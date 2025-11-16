# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================
import torch
import torch.nn.functional as F
import argparse
import os
from omegaconf import OmegaConf

from model.ode_regression import ODERegression
from utils.dataset import ODERegressionLMDBDataset
from utils.misc import set_seed

# ============================================================================
# 2. MONKEY PATCH FOR ODEREGRESSION
# ============================================================================
# Fix signature mismatch between ODERegression and BaseModel
_original_init = ODERegression._initialize_models
def _patched_initialize_models(self, args, device=None):
    result = _original_init(self, args)
    # Add scheduler support for warp_denoising_step
    if not hasattr(self, 'scheduler'):
        self.scheduler = self.generator.get_scheduler()
    return result
ODERegression._initialize_models = _patched_initialize_models


# ============================================================================
# 3. MAIN VALIDATION FUNCTION
# ============================================================================
def calculate_reconstruction_loss_for_checkpoint(
    config_path: str, 
    checkpoint_path: str, 
    batch: dict,
    checkpoint_name: str,
    validation_seed: int
):
    """Validates checkpoint and returns MSE reconstruction loss."""
    print(f"\n{'='*80}")
    print(f"Validating {checkpoint_name}")
    print(f"{'='*80}")
    
    # --- 3.1 Configuration and Model Initialization ---
    config = OmegaConf.load(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    model = ODERegression(config, device=device)
    
    # --- 3.2 Checkpoint Loading ---
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats (DMD uses generator_ema, ODE uses generator)
    if 'generator_ema' in checkpoint_data:
        state_dict = checkpoint_data['generator_ema']
    elif 'generator' in checkpoint_data:
        state_dict = checkpoint_data['generator']
    else:
        state_dict = checkpoint_data
    
    model.generator.load_state_dict(state_dict, strict=True)
    
    # --- 3.3 Device Setup --- one gpu for now
    model.generator.to(device, dtype=model.dtype)
    model.text_encoder.to(device, dtype=model.dtype)
    
    if hasattr(model, 'denoising_step_list'):
        model.denoising_step_list = model.denoising_step_list.to(device)
    
    model.eval()

    # --- 3.4 Batch Preparation ---
    text_prompts = batch["prompts"]
    ode_latent = batch["ode_latent"].to(device, dtype=model.dtype)
    ground_truth_latent = ode_latent[:, -1].detach()

    # --- 3.5 Inference and Loss Calculation ---
    with torch.no_grad():
        # Set seed immediately before timestep sampling for perfect reproducibility
        set_seed(validation_seed)
        
        # Text encoding
        conditional_dict = model.text_encoder(text_prompts=text_prompts)
        
        # Sample random intermediate timestep (the actual training objective)
        noisy_input, timestep = model._prepare_generator_input(ode_latent=ode_latent)
        
        # Generator forward pass
        _, prediction = model.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Calculate reconstruction loss
        l2_loss = F.mse_loss(prediction, ground_truth_latent, reduction="mean")

    print(f"\n--- {checkpoint_name} Result ---")
    print(f"MSE Loss: {l2_loss.item():.6f}")
    
    per_sample_loss = F.mse_loss(prediction, ground_truth_latent, reduction='none')
    per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
    print(f"Per-sample: {[f'{x:.6f}' for x in per_sample_loss.tolist()]}")
    
    return l2_loss.item()


# ============================================================================
# 4. MAIN
# ============================================================================
if __name__ == "__main__":
    # --- 4.1 Argument Parsing ---
    parser = argparse.ArgumentParser(description="Compare ODE and DMD checkpoint reconstruction loss")
    parser.add_argument("--ode_config", type=str, default="configs/ode_validation.yaml")
    parser.add_argument("--dmd_config", type=str, default="configs/dmd_validation.yaml")
    parser.add_argument("--ode_checkpoint", type=str, default="checkpoints/ode_init.pt")
    parser.add_argument("--dmd_checkpoint", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    # Use a constant validation seed
    VAL_SEED = 42
    
    # --- 4.2 Load Data Once (For Perfect Reproducibility) ---
    print(f"\nLoading data from: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    
    # Set seed for reproducible data loading
    set_seed(VAL_SEED)
    
    max_samples = max(100, args.batch_size * 10)
    dataset = ODERegressionLMDBDataset(args.data_path, max_pair=max_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=min(4, args.batch_size),
        pin_memory=torch.cuda.is_available()
    )
    
    try:
        batch = next(iter(dataloader))
        print(f"Loaded batch with {len(batch['prompts'])} samples")
    except StopIteration:
        print("Error: Dataset is empty.")
        exit(1)
    
    # --- 4.3 Validate ODE Checkpoint ---
    ode_loss = calculate_reconstruction_loss_for_checkpoint(
        args.ode_config, args.ode_checkpoint, batch,
        "ODE Init Checkpoint", VAL_SEED
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # --- 4.4 Validate DMD Checkpoint ---
    dmd_loss = calculate_reconstruction_loss_for_checkpoint(
        args.dmd_config, args.dmd_checkpoint, batch,
        "DMD Checkpoint", VAL_SEED
    )
    
    # --- 4.5 Comparison Summary ---
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if ode_loss and dmd_loss:
        improvement = ode_loss - dmd_loss
        improvement_pct = (improvement / ode_loss) * 100
        
        print(f"ODE Init: {ode_loss:.6f}")
        print(f"DMD:      {dmd_loss:.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        
        if dmd_loss < ode_loss:
            print(f"✓ DMD is better (expected)")
        else:
            print(f"✗ DMD is worse (unexpected)")
    else:
        print("Validation failed")
    
    print(f"{'='*80}\n")


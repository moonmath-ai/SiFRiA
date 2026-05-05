import argparse
import torch
import os
import glob
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="Generate videos for DMD checkpoints")
parser.add_argument("--checkpoint_dir", type=str, default="/data/karthik_data/expt_2_1_1_3b_dmd_14b_teacher")
parser.add_argument("--config", type=str, default="configs/self_forcing_dmd_modified.yaml")
parser.add_argument("--prompt_file", type=str, default="prompts/MovieGenVideoBench_extended_diagnostic_24.txt")
parser.add_argument("--output_folder", type=str, default="videos/compare_dmd")
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights")
args = parser.parse_args()

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

print(f'Device: {device}')
print(f'Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

# Create output directory
os.makedirs(args.output_folder, exist_ok=True)

# Read all prompts from file
with open(args.prompt_file, 'r') as f:
    lines = f.readlines()
    prompts = []
    for line in lines:
        line = line.strip()
        if line:
            prompts.append(line)
    
    if not prompts:
        raise ValueError("No prompts found in file")

print(f"\n{'='*80}")
print(f"Found {len(prompts)} prompts to process")
print(f"{'='*80}\n")

# Find all checkpoint directories
checkpoint_dirs = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_model_*")))

if not checkpoint_dirs:
    print(f"No checkpoints found in {args.checkpoint_dir}")
    exit(1)

print(f"Found {len(checkpoint_dirs)} checkpoints:")
for ckpt_dir in checkpoint_dirs:
    print(f"  - {os.path.basename(ckpt_dir)}")
print()

# Generate video for each prompt and checkpoint
for prompt_idx, prompt in enumerate(prompts, 1):
    print(f"\n{'='*80}")
    print(f"PROMPT {prompt_idx}/{len(prompts)}: {prompt[:100]}...")
    print(f"{'='*80}\n")
    
    for ckpt_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {ckpt_dir} - model.pt not found")
            continue
        
        checkpoint_name = os.path.basename(ckpt_dir)
        iteration = checkpoint_name.split("_")[-1]
        
        print(f"\n{'='*80}")
        print(f"Generating video for {checkpoint_name}")
        print(f"{'='*80}")
        
        # Load config
        config = OmegaConf.load(args.config)
        default_config = OmegaConf.load("configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)
        
        # Initialize pipeline
        pipeline = CausalInferencePipeline(config, device=device)
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if args.use_ema and 'generator_ema' in state_dict:
            weights = state_dict['generator_ema']
            print("Loaded EMA weights")
        elif 'generator' in state_dict:
            weights = state_dict['generator']
            print("Loaded generator weights")
        else:
            weights = state_dict
            print("Loaded checkpoint weights")
        
        # Strip FSDP wrapper prefix if present
        if any(k.startswith('model._fsdp_wrapped_module.') for k in weights.keys()):
            print("Stripping FSDP wrapper prefix...")
            weights = {k.replace('model._fsdp_wrapped_module.', 'model.'): v for k, v in weights.items()}
        
        pipeline.generator.load_state_dict(weights)
        
        # Move to device and dtype
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        else:
            pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        
        # Set seed for reproducibility
        set_seed(args.seed)
        
        # Generate noise
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], 
            device=device, 
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(args.seed)
        )
        
        # Generate video
        print(f"Generating {args.num_output_frames} frames...")
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            return_latents=True,
            initial_latent=None,
            low_memory=low_memory,
        )
        
        # Process video
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = 255.0 * video
        
        # Clear VAE cache
        pipeline.vae.model.clear_cache()
        
        # Save video with prompt index
        output_path = os.path.join(args.output_folder, f'prompt{prompt_idx:02d}_iter_{iteration}.mp4')
        write_video(output_path, video[0], fps=16)
        print(f"Saved: {output_path}")
        
        # Clean up
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("VIDEO GENERATION COMPLETE")
print(f"{'='*80}")
print(f"Videos saved in: {args.output_folder}")
print(f"Generated {len(prompts)} prompts x {len(checkpoint_dirs)} checkpoints = {len(prompts) * len(checkpoint_dirs)} videos")
print(f"Naming format: prompt[01-{len(prompts):02d}]_iter_[iteration].mp4")
print(f"{'='*80}\n")


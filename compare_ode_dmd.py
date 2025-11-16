import argparse
import torch
import os
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="Compare ODE vs DMD checkpoint video generation")
parser.add_argument("--ode_config", type=str, default="configs/wan_causal_ode.yaml")
parser.add_argument("--dmd_config", type=str, default="configs/self_forcing_dmd.yaml")
parser.add_argument("--ode_checkpoint", type=str, default="checkpoints/ode_init.pt")
parser.add_argument("--dmd_checkpoint", type=str, default="checkpoints/self_forcing_dmd.pt")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt text file")
parser.add_argument("--output_folder", type=str, default="videos/ode_vs_dmd")
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--use_ema", action="store_true", help="Use EMA weights for DMD")
args = parser.parse_args()

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

print(f'Device: {device}')
print(f'Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

# Create output directory
os.makedirs(args.output_folder, exist_ok=True)

# Read prompt from file (first non-empty line)
with open(args.prompt_file, 'r') as f:
    lines = f.readlines()
    prompt = None
    for line in lines:
        line = line.strip()
        if line:
            prompt = line
            break
    
    if prompt is None:
        raise ValueError("No prompt found in file")

print(f"\n{'='*80}")
print(f"Prompt: {prompt[:200]}...")
print(f"{'='*80}\n")

# Generate video with both checkpoints
for checkpoint_name, config_path, checkpoint_path in [
    ("ODE", args.ode_config, args.ode_checkpoint),
    ("DMD", args.dmd_config, args.dmd_checkpoint)
]:
    print(f"\n{'='*80}")
    print(f"Generating with {checkpoint_name} checkpoint")
    print(f"{'='*80}")
    
    # Load config
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    # Initialize pipeline
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load checkpoint
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_name == "DMD" and args.use_ema and 'generator_ema' in state_dict:
            pipeline.generator.load_state_dict(state_dict['generator_ema'])
            print("Loaded EMA weights for DMD")
        elif 'generator_ema' in state_dict:
            pipeline.generator.load_state_dict(state_dict['generator_ema'])
            print("Loaded generator_ema weights")
        elif 'generator' in state_dict:
            pipeline.generator.load_state_dict(state_dict['generator'])
            print("Loaded generator weights")
        else:
            pipeline.generator.load_state_dict(state_dict)
            print("Loaded checkpoint weights")
    
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
    
    # Save video
    output_path = os.path.join(args.output_folder, f'{checkpoint_name.lower()}_generation.mp4')
    write_video(output_path, video[0], fps=16)
    print(f"Saved: {output_path}")
    
    # Clean up
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("COMPARISON COMPLETE")
print(f"{'='*80}")
print(f"Videos saved in: {args.output_folder}")
print(f"  - ode_generation.mp4")
print(f"  - dmd_generation.mp4")
print(f"{'='*80}\n")


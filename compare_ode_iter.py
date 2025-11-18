import argparse
import torch
import os
import glob
import re
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="Compare ODE checkpoints across training iterations")
parser.add_argument("--ode_config", type=str, default="configs/wan_causal_ode.yaml")
parser.add_argument("--checkpoint_dir", type=str, default="/root/karthik/ode_checkpoints")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt text file")
parser.add_argument("--output_folder", type=str, default="videos/compare_ode")
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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

# Find all checkpoints
checkpoint_pattern = os.path.join(args.checkpoint_dir, "checkpoint_model_*/model.pt")
checkpoint_paths = sorted(glob.glob(checkpoint_pattern))

if not checkpoint_paths:
    raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")

print(f"Found {len(checkpoint_paths)} checkpoints:")
for cp in checkpoint_paths:
    print(f"  - {cp}")
print("")

# Extract iteration numbers and sort
checkpoints = []
for cp_path in checkpoint_paths:
    # Extract iteration number from path like "checkpoint_model_000200/model.pt"
    match = re.search(r'checkpoint_model_(\d+)', cp_path)
    if match:
        iteration = int(match.group(1))
        checkpoints.append((iteration, cp_path))

# Sort by iteration
checkpoints.sort(key=lambda x: x[0])

# Load config once (same for all checkpoints)
config = OmegaConf.load(args.ode_config)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Generate video for each checkpoint
generated_videos = []

for iteration, checkpoint_path in checkpoints:
    print(f"\n{'='*80}")
    print(f"Generating with checkpoint at iteration {iteration}")
    print(f"Path: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Initialize pipeline
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if 'generator_ema' in state_dict:
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
    
    # Set seed for reproducibility (same seed for all checkpoints)
    set_seed(args.seed)
    
    # Generate noise (same noise for all checkpoints)
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
    output_filename = f'iter_{iteration:06d}.mp4'
    output_path = os.path.join(args.output_folder, output_filename)
    write_video(output_path, video[0], fps=16)
    print(f"Saved: {output_path}")
    generated_videos.append((iteration, output_filename))
    
    # Clean up
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("GENERATION COMPLETE")
print(f"{'='*80}")
print(f"Videos saved in: {args.output_folder}")
for iteration, filename in generated_videos:
    print(f"  - Iteration {iteration:6d}: {filename}")
print(f"{'='*80}\n")


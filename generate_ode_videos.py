#!/usr/bin/env python3
"""
Generate videos from ODE checkpoint for visual inspection
"""
import argparse
import torch
import os
import glob
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.dataset import TextDataset
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="Generate videos from ODE checkpoint")
parser.add_argument("--config", type=str, default="configs/wan_causal_ode_modified.yaml")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint folders")
parser.add_argument("--prompt_file", type=str, default="prompts/test_prompts.txt", help="Path to prompt text file")
parser.add_argument("--output_folder", type=str, default="videos/ode_inspection")
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--num_prompts", type=int, default=4, help="Number of prompts to generate")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--checkpoint_step", type=str, default=None, help="Specific checkpoint step (e.g., 001000), or leave empty for all")
args = parser.parse_args()

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

print(f'Device: {device}')
print(f'Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

# Create output directory
os.makedirs(args.output_folder, exist_ok=True)

# Load prompts from file
dataset = TextDataset(prompt_path=args.prompt_file)
num_prompts_available = len(dataset)
num_prompts_to_use = min(args.num_prompts, num_prompts_available)

print(f"\nLoaded {num_prompts_available} prompts from {args.prompt_file}")
print(f"Will generate videos for {num_prompts_to_use} prompts")

prompts = []
for i in range(num_prompts_to_use):
    prompt_dict = dataset[i]
    prompts.append(prompt_dict["prompts"])
    print(f"  {i+1}. {prompt_dict['prompts'][:100]}...")

# Find checkpoints
if args.checkpoint_step:
    checkpoint_dirs = [os.path.join(args.checkpoint_dir, f"checkpoint_model_{args.checkpoint_step}")]
else:
    checkpoint_dirs = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_model_*")))

if not checkpoint_dirs:
    print(f"No checkpoints found in {args.checkpoint_dir}")
    exit(1)

print(f"\nFound {len(checkpoint_dirs)} checkpoint(s):")
for ckpt_dir in checkpoint_dirs:
    print(f"  - {os.path.basename(ckpt_dir)}")
print()

# Generate videos for each checkpoint
for ckpt_dir in checkpoint_dirs:
    checkpoint_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {ckpt_dir} - model.pt not found")
        continue
    
    checkpoint_name = os.path.basename(ckpt_dir)
    iteration = checkpoint_name.split("_")[-1]
    
    print(f"\n{'='*80}")
    print(f"Generating videos for {checkpoint_name}")
    print(f"{'='*80}")
    
    # Load config
    config = OmegaConf.load(args.config)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
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
    
    # Create checkpoint-specific output folder
    checkpoint_output_folder = os.path.join(args.output_folder, f"step_{iteration}")
    os.makedirs(checkpoint_output_folder, exist_ok=True)
    
    # Generate videos for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{num_prompts_to_use}] Generating: {prompt[:100]}...")
        
        # Set seed for reproducibility
        seed = args.seed + prompt_idx
        set_seed(seed)
        
        # Generate noise
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], 
            device=device, 
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(seed)
        )
        
        # Generate video
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
        output_path = os.path.join(checkpoint_output_folder, f'prompt_{prompt_idx:02d}.mp4')
        write_video(output_path, video[0], fps=16)
        print(f"  Saved: {output_path}")
        
        # Also save prompt to txt file
        prompt_file_path = os.path.join(checkpoint_output_folder, f'prompt_{prompt_idx:02d}.txt')
        with open(prompt_file_path, 'w') as f:
            f.write(prompt)
    
    # Clean up
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("VIDEO GENERATION COMPLETE")
print(f"{'='*80}")
print(f"Videos saved in: {args.output_folder}")
print(f"{'='*80}\n")


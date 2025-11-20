import argparse
import torch
import os
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="Batched Text-to-Video Inference")
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, required=True, help="Path to the prompts text file")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of frames to generate per video")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for inference (number of prompts to process simultaneously)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--profile", action="store_true",
                    help="Enable PyTorch profiler for performance analysis")
parser.add_argument("--profile_output", type=str, default="profiler_results",
                    help="Output folder for profiler results")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Custom collate function to handle batching of text prompts
def custom_collate_fn(batch):
    """
    Collate function that properly batches text prompts and indices.
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Dictionary with batched prompts and indices
    """
    prompts = [item['prompts'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    result = {
        'prompts': prompts,
        'idx': indices,
    }
    
    # Handle extended prompts if they exist
    if 'extended_prompts' in batch[0]:
        extended_prompts = [item['extended_prompts'] for item in batch]
        result['extended_prompts'] = extended_prompts
    
    return result


# Create dataset for T2V
dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)

# Use the custom collate function and batch_size from args
dataloader = DataLoader(
    dataset, 
    batch_size=args.batch_size, 
    sampler=sampler, 
    num_workers=0, 
    drop_last=False,
    collate_fn=custom_collate_fn
)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    if args.profile:
        os.makedirs(args.profile_output, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

print(f"Processing with batch size: {args.batch_size}")
print(f"Total batches: {len(dataloader)}")
print(f"Expected total videos: {num_prompts * args.num_samples}")
print("\n" + "="*60)

# Start timing
start_time = time.time()
batch_times = []
total_videos_generated = 0

# Setup profiler if enabled
profiler_context = None
if args.profile and local_rank == 0:
    profiler_context = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler_context.__enter__()
    print(f"Profiling enabled. Results will be saved to: {args.profile_output}")
    print("TensorBoard trace will be generated. View with: tensorboard --logdir={args.profile_output}")

for batch_idx, batch_data in enumerate(tqdm(dataloader, disable=(local_rank != 0))):
    batch_start_time = time.time()
    
    # Get the batch size for this iteration (might be smaller for the last batch)
    current_batch_size = len(batch_data['idx'])
    
    # For text-to-video, batch contains text prompts
    prompts_batch = batch_data['prompts']
    
    # Use extended prompts if available (these are usually better)
    if 'extended_prompts' in batch_data:
        prompts_batch = batch_data['extended_prompts']
    
    # Repeat each prompt for num_samples (if generating multiple variations)
    prompts = []
    for prompt in prompts_batch:
        prompts.extend([prompt] * args.num_samples)
    
    # No initial latent for T2V (start from pure noise)
    initial_latent = None
    
    with record_function("generate_noise"):
        # Generate random noise for each video
        sampled_noise = torch.randn(
            [current_batch_size * args.num_samples, args.num_output_frames, 16, 60, 104], 
            device=device, 
            dtype=torch.bfloat16
        )
    
    with record_function("pipeline_inference"):
        # Generate videos for all prompts in the batch
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )
    
    with record_function("postprocess_video"):
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        
        # Final output video
        video_output = 255.0 * current_video
        
        # Clear VAE cache
        pipeline.vae.model.clear_cache()
    
    with record_function("save_videos"):
        # Save each video in the batch
        for i, idx in enumerate(batch_data['idx']):
            # Only save if the current prompt is not a dummy prompt
            if idx < num_prompts:
                model = "regular" if not args.use_ema else "ema"
                # Get the original prompt for filename
                prompt = batch_data['prompts'][i]
                
                # Save each sample for this prompt
                for seed_idx in range(args.num_samples):
                    video_idx = i * args.num_samples + seed_idx
                    
                    if args.save_with_index:
                        output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
                    else:
                        # Sanitize prompt for filename
                        safe_prompt = prompt[:100].replace('/', '_').replace('\\', '_')
                        output_path = os.path.join(args.output_folder, f'{safe_prompt}-{seed_idx}.mp4')
                    
                    write_video(output_path, video_output[video_idx], fps=16)
    
    # Record batch timing
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    batch_times.append(batch_duration)
    total_videos_generated += current_batch_size * args.num_samples
    
    if local_rank == 0:
        videos_in_batch = current_batch_size * args.num_samples
        time_per_video = batch_duration / videos_in_batch
        print(f"Batch {batch_idx + 1}/{len(dataloader)} completed in {batch_duration:.2f}s "
              f"({time_per_video:.2f}s per video, {current_batch_size} prompts)")
    
    # Step profiler if enabled
    if profiler_context is not None:
        profiler_context.step()

# Stop profiler if enabled
if profiler_context is not None:
    profiler_context.__exit__(None, None, None)
    print(f"\nProfiler results saved to: {args.profile_output}")
    print(f"View with TensorBoard: tensorboard --logdir={args.profile_output}")

# Calculate and display timing statistics
end_time = time.time()
total_duration = end_time - start_time

if local_rank == 0:
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"Total inference time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Total videos generated: {total_videos_generated}")
    print(f"Average time per video: {total_duration/total_videos_generated:.2f}s")
    print(f"Throughput: {total_videos_generated/total_duration:.2f} videos/second")
    print(f"Total frames generated: {total_videos_generated * args.num_output_frames}")
    print(f"Frame generation rate: {(total_videos_generated * args.num_output_frames)/total_duration:.2f} frames/second")
    
    if len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"\nAverage batch time: {avg_batch_time:.2f}s")
        print(f"Min batch time: {min(batch_times):.2f}s")
        print(f"Max batch time: {max(batch_times):.2f}s")
    
    print("="*60)
    print(f"Inference complete! Videos saved to: {args.output_folder}")
    print("="*60)


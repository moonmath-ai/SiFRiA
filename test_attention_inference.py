"""
Full script to track Flash and Flex Attention invocations during Autoregressive (AR) 
video generation and save the resulting video.

"""

import torch
import os
import sys
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from torchvision.io import write_video
from einops import rearrange
from typing import Dict, Any, Optional, List

# --- GLOBAL TRACKERS ---
ATTENTION_CALLS: Dict[str, int] = {"flex": 0, "flash_cross": 0, "flash_self": 0} 
ORIGINAL_FUNCS: Dict[str, Any] = {}
CAUSAL_MODEL_MODULE = None
ATTENTION_MODULE = None


def setup_attention_tracking():
    """Applies monkey patches to track attention calls."""
    global CAUSAL_MODEL_MODULE, ATTENTION_MODULE
    
    # 1. --- FlexAttention ---
    try:
        import wan.modules.causal_model as causal_model
        CAUSAL_MODEL_MODULE = causal_model
        
        def tracked_flex_attention(*args, **kwargs):
            ATTENTION_CALLS["flex"] += 1
            return ORIGINAL_FUNCS["flex"](*args, **kwargs)

        ORIGINAL_FUNCS["flex"] = causal_model.flex_attention
        causal_model.flex_attention = tracked_flex_attention
        print("Installed FlexAttention tracker (Masked/Training Path)")
    except Exception as e:
        print(f" Could not install FlexAttention tracker: {e}")

    # --- 2. Generic Attention Wrapper (Self-Attn) & Primitive (Cross-Attn) ---
    try:
        import wan.modules.attention as attn_module
        ATTENTION_MODULE = attn_module
        
        # Tracker for the KV Lookup Wrapper (Self-Attn, for diagnostics)
        def tracked_attention_wrapper(*args, **kwargs):
            ATTENTION_CALLS["flash_self"] += 1
            return ORIGINAL_FUNCS["flash_self"](*args, **kwargs) 

        # Tracker for the Primitive Kernel (Used by Cross-Attn, captures total)
        def tracked_flash_primitive(*args, **kwargs):
            ATTENTION_CALLS["flash_cross"] += 1 
            return ORIGINAL_FUNCS["flash_cross"](*args, **kwargs)

        ORIGINAL_FUNCS["flash_self"] = attn_module.attention
        attn_module.attention = tracked_attention_wrapper
        
        ORIGINAL_FUNCS["flash_cross"] = attn_module.flash_attention
        attn_module.flash_attention = tracked_flash_primitive
        
        print("Installed Flash Wrapper tracker [Self-Attn Count]")
        print("Installed Flash Primitive tracker [Cross-Attn Kernel]")
    except Exception as e:
        print(f"Could not install Flash trackers: {e}")


def restore_original_functions():
    """Restores the original attention functions."""
    if ATTENTION_MODULE and "flash_cross" in ORIGINAL_FUNCS:
        ATTENTION_MODULE.flash_attention = ORIGINAL_FUNCS["flash_cross"]
    if ATTENTION_MODULE and "flash_self" in ORIGINAL_FUNCS:
        ATTENTION_MODULE.attention = ORIGINAL_FUNCS["flash_self"]
    if CAUSAL_MODEL_MODULE and "flex" in ORIGINAL_FUNCS:
        CAUSAL_MODEL_MODULE.flex_attention = ORIGINAL_FUNCS["flex"]
    print("\n[INFO] Restored original attention functions.")


def main():
    setup_attention_tracking()
    print("=" * 80)
    print("ATTENTION MECHANISM TEST - Autoregressive Inference")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    torch.set_grad_enabled(False)
       
    config_path = "configs/self_forcing_dmd.yaml"
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    if not hasattr(config, 'denoising_step_list'):
        print("Config not suitable for causal pipeline test.")
        return
        
    pipeline = CausalInferencePipeline(config, device=device)
    checkpoint_path = "checkpoints/self_forcing_dmd.pt"
    
    try:
        print(f" Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = 'generator_ema' if 'generator_ema' in state_dict else 'generator'
        pipeline.generator.load_state_dict(state_dict[key], strict=False)
        print(f" Loaded {key} weights")
    except Exception as e:
        print(f"  Checkpoint loading failed or not found: {e}")
        print("   Continuing with uninitialized weights (for attention test only)")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)
    
    print(f"\nPipeline initialized successfully. Starting inference.")

    # Test prompt 
    prompt = "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
    num_frames = 21
    noise = torch.randn(
        [1, num_frames, 16, 60, 104], 
        device=device, 
        dtype=torch.bfloat16
    )
    
    print(f"\n Starting generation (this will trigger attention calls)...\n")
    print("-" * 80)
    
    # Reset counters
    ATTENTION_CALLS["flex"] = 0
    ATTENTION_CALLS["flash_cross"] = 0
    ATTENTION_CALLS["flash_self"] = 0
    
    # Run inference
    video = None
    try:
        video, latents = pipeline.inference(
            noise=noise, text_prompts=[prompt], return_latents=True, low_memory=False,
        )
        
        print("-" * 80)
        print(f"\n Generation complete!")
        print(f"   Output video shape: {video.shape}")
        
        # --- VIDEO SAVING ---
        output_dir = "videos/attention_tests"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "attention_test_video.mp4")
        video_to_save = video[0].cpu() 
        video_to_save = (video_to_save * 255).to(torch.uint8)
        video_to_save = rearrange(video_to_save, 'f c h w -> f h w c')
        write_video(output_path, video_to_save, fps=16)
        print(f"\n Video saved to: {output_path}")
        
    except Exception as e:
        print(f"\n Error during generation: {e}")
        # traceback.print_exc()
        
    finally:
        restore_original_functions()
    
    # Print results
    print("\n" + "=" * 80)
    print("ATTENTION MECHANISM ANALYSIS")
    print("=" * 80)
    
    flex_calls = ATTENTION_CALLS['flex']
    flash_self_calls = ATTENTION_CALLS['flash_self']
    flash_cross_calls = ATTENTION_CALLS['flash_cross']
    
    
    print(f" FlexAttention (Masked/Training Path) calls: {flex_calls}")
    print(f"Flash (Inference Wrapper) calls [SELF-ATTN KV]: {flash_self_calls}")
    print(f"Flash (Primitive Kernel) calls [CROSS-ATTN]: {flash_cross_calls}")

if __name__ == "__main__":
    main()
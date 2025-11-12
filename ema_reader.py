import torch
import os
from typing import Dict, Any

def save_ema_contents_to_file(checkpoint_path: str, output_filename: str = 'ema_contents.txt') -> None:
    """
    Loads the checkpoint, extracts the 'generator_ema' dictionary, and saves 
    a detailed list of all parameters (keys, shapes, dtypes) to a text file.
    """
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at path: {checkpoint_path}")
        return

    print(f"-> Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the dictionary structure to CPU
        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint file. Exception: {e}")
        return

    # 1. Validate EMA key presence
    if 'generator_ema' not in checkpoint or not isinstance(checkpoint['generator_ema'], dict):
        print("ERROR: 'generator_ema' key not found or is not a dictionary. Cannot proceed.")
        return
        
    ema_dict = checkpoint['generator_ema']
    
    # 2. Write contents to the file
    with open(output_filename, 'w') as f:
        
        # Write Header
        f.write(f"--- Contents of 'generator_ema' (Total Parameters: {len(ema_dict)}) ---\n")
        f.write("----------------------------------------------------------------------\n\n")

        # Track total parameters and memory size
        total_params = 0
        
        for key, tensor in ema_dict.items():
            if isinstance(tensor, torch.Tensor):
                # Calculate number of elements and data type
                num_elements = tensor.numel()
                total_params += num_elements
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype).replace('torch.', '')
                
                # Write entry to file
                f.write(f"Key: {key}\n")
                f.write(f"  Shape: {shape}\n")
                f.write(f"  DType: {dtype}\n")
                f.write(f"  Size: {num_elements:,} elements\n")
                f.write("-" * 20 + "\n")
            else:
                # Handle non-tensor entries (like simple metrics, though unlikely here)
                f.write(f"Key: {key}\n")
                f.write(f"  Value: {tensor} (Type: {type(tensor).__name__})\n")
                f.write("-" * 20 + "\n")
        
        # Write Footer Summary
        f.write("\n======================================================================\n")
        f.write(f"SUMMARY: Total number of tensor elements in EMA: {total_params:,}\n")
        f.write(f"Output saved to: {output_filename}\n")


# --- Execution ---
CHECKPOINT_FILE = 'checkpoints/self_forcing_dmd.pt' 
OUTPUT_FILE = 'ema_contents.txt'

save_ema_contents_to_file(CHECKPOINT_FILE, OUTPUT_FILE)

print(f"\n✅ Successfully generated the file: **{OUTPUT_FILE}**")
print(f"You can now open this file to review all {len(torch.load(CHECKPOINT_FILE, map_location='cpu')['generator_ema'])} EMA parameters.")
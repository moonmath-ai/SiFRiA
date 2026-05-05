#!/usr/bin/env python3
# use wandb logs to print loss values
"""
Check ODE training losses from wandb
"""
import wandb

# Login to wandb
wandb.login(key="bf3087c2d3a5361a4b222a9c1529e9017ddb630a")

# Get the run
api = wandb.Api()
entity = "lnm-"
project = "self_forcing_ode_2.1_1.3b_data_gs_6_shift_5"

print(f"Fetching runs from {entity}/{project}...")
runs = api.runs(f"{entity}/{project}")

# Find the most recent ODE training run
latest_run = None
for run in runs:
    latest_run = run
    break

if not latest_run:
    print("No ODE training run found!")
    exit(1)

print(f"\nFound run: {latest_run.name} (ID: {latest_run.id})")
print(f"URL: {latest_run.url}")

# Get the history (all logged metrics)
print("\nFetching training history...")
history = latest_run.scan_history()

# Collect all data points
all_data = []
for row in history:
    if 'generator_loss' in row and '_step' in row:
        all_data.append(row)

if not all_data:
    print("\nNo generator_loss data found!")
    exit(1)

print(f"Found {len(all_data)} data points")

# Sort all data by step
all_data_sorted = sorted(all_data, key=lambda x: x.get('_step', 0))

# Show the last few logged steps
if all_data_sorted:
    print(f"\nLast 5 logged steps:")
    for data in all_data_sorted[-5:]:
        step = data.get('_step', 0)
        loss = data.get('generator_loss', None)
        if loss is not None:
            print(f"  Step {step}: loss {loss:.5f}")
    last_step = all_data_sorted[-1].get('_step', 0)
    last_loss = all_data_sorted[-1].get('generator_loss', None)
    if last_loss is not None:
        print(f"\nNote: Training starts at step 0, so step {last_step} = {last_step+1} iterations completed")
        print(f"      Step {last_step} (loss: {last_loss:.5f}) corresponds to checkpoint_model_003000")
    else:
        print(f"\nNote: Step {last_step} has no loss logged")

# Compute EMA for ALL data points
ema_decay = 0.9
ema_loss = None
ema_by_step = {}

for data in all_data_sorted:
    step = data.get('_step', 0)
    gen_loss = data.get('generator_loss', None)
    
    if gen_loss is not None:
        if ema_loss is None:
            ema_loss = gen_loss
        else:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * gen_loss
        
        ema_by_step[step] = ema_loss

# Get checkpoint steps (including initial step 0 and every 250 iterations)
# Note: Training starts at step 0, so checkpoint_model_003000 is saved at step 2999
checkpoint_steps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 2999]

print("\n" + "="*80)
print("ODE Training Loss at Each Checkpoint")
print("="*80)
print(f"{'Step':<10} {'Raw Loss':<15} {'EMA Loss (0.9)':<20} {'Note':<30}")
print("-"*80)

checkpoint_data = []

for target_step in checkpoint_steps:
    # Special handling for step 2999 (checkpoint_model_003000) - find the last logged step WITH a loss value
    if target_step == 2999 and all_data_sorted:
        # Find the last data point that has a valid loss
        closest_data = None
        for data in reversed(all_data_sorted):
            if data.get('generator_loss') is not None:
                closest_data = data
                break
    else:
        # Find the data point closest to this checkpoint step (within 50 steps tolerance)
        closest_data = None
        min_diff = float('inf')
        
        for data in all_data_sorted:  # Use sorted data
            step = data.get('_step', 0)
            diff = abs(step - target_step)
            if diff < min_diff and diff <= 50:  # Increased tolerance to 50 steps
                min_diff = diff
                closest_data = data
    
    if closest_data:
        actual_step = closest_data.get('_step', target_step)
        gen_loss = closest_data.get('generator_loss', None)
        ema_loss = ema_by_step.get(actual_step, None)
        
        if gen_loss is not None:
            if ema_loss is None:
                # For steps where EMA isn't computed yet, show just raw loss
                note = "(Untrained loss - no ckpt)" if actual_step == 0 else ""
                print(f"{actual_step:<10} {gen_loss:<15.5f} {'N/A':<20} {note:<30}")
            else:
                note = "(Untrained loss - no ckpt)" if actual_step == 0 else ""
                if actual_step != target_step:
                    note = f"(closest to {target_step})"
                print(f"{actual_step:<10} {gen_loss:<15.5f} {ema_loss:<20.5f} {note:<30}")
            
            # Only add to checkpoint_data if it's not step 0 (since there's no checkpoint)
            if actual_step != 0 and ema_loss is not None:
                checkpoint_data.append({
                    'step': actual_step,
                    'generator_loss': gen_loss,
                    'ema_loss': ema_loss
                })
        else:
            print(f"{target_step:<10} {'N/A':<15} {'N/A':<20}")
    else:
        # No data found 
        print(f"{target_step:<10} {'N/A':<15} {'N/A':<20} (no data found)")

# Find the checkpoint with lowest EMA loss
if checkpoint_data:
    best_checkpoint_raw = min(checkpoint_data, key=lambda x: x['generator_loss'])
    best_checkpoint_ema = min(checkpoint_data, key=lambda x: x['ema_loss'])
    
    print("\n" + "="*80)
    print(f"✅ BEST CHECKPOINT (by EMA): Step {int(best_checkpoint_ema['step']):06d}")
    print(f"   Raw Loss: {best_checkpoint_ema['generator_loss']:.5f}")
    print(f"   EMA Loss: {best_checkpoint_ema['ema_loss']:.5f}")
    print(f"   Path: /data/karthik_data/expt_2_1_1_3b_ode/checkpoint_model_{int(best_checkpoint_ema['step']):06d}/")
    
    if best_checkpoint_raw['step'] != best_checkpoint_ema['step']:
        print(f"\n   Note: Best by raw loss was Step {int(best_checkpoint_raw['step']):06d} ({best_checkpoint_raw['generator_loss']:.5f})")
    print("="*80)
else:
    print("\nCould not determine best checkpoint!")


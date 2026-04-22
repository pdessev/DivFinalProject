import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # ADD THIS LINE
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random

# Import your custom modules
from dataset import FullVideoDataset
from CNNv1 import CNNEncoder, BiConvLSTM, TimeConditionedDecoder
from warping import WarpingModule
from loss import VFICombinedLoss

# Add scaler to the arguments
def train_one_epoch(encoder, lstm, decoder, warping_module, criterion, optimizer, dataloader, device, scaler, subsample_factor=10):
    encoder.train()
    lstm.train()
    decoder.train()
    
    running_loss = 0.0
    all_params = list(encoder.parameters()) + list(lstm.parameters()) + list(decoder.parameters())
    
    for batch_idx, (context_frames, all_frames) in enumerate(dataloader):
        context_frames = context_frames.to(device)
        all_frames = all_frames.to(device)
        
        optimizer.zero_grad()
            
        seq_len = context_frames.size(1)
        
        lstm_inputs = []
        skips_list = []
        
        # --- NEW: Wrap the forward pass in autocast ---
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            
            # 1. Run the FULL sequence through the CNN Encoder
            for i in range(seq_len):
                features, f1, f2, f3 = encoder(context_frames[:, i])
                lstm_inputs.append(features)
                skips_list.append([f1, f2, f3])
                
            lstm_sequence = torch.stack(lstm_inputs, dim=1)
            
            # 2. Process the sequence through the BiConvLSTM
            lstm_outputs = lstm(lstm_sequence)
            
            # 3. Choose a random interval
            interval = random.randint(0, seq_len - 2)
            
            frame_0 = context_frames[:, interval]
            frame_1 = context_frames[:, interval + 1]
            target_lstm_state = lstm_outputs[:, interval]
            skips = skips_list[interval]
            
            interval_loss = 0.0
            num_frames_predicted = 0
            
            # 4. Generate the missing frames
            for offset in range(1, subsample_factor):
                t_val = offset / float(subsample_factor)
                
                # Fixed: Reverted back to just using 'interval' since truncation is gone
                gt_idx = (interval * subsample_factor) + offset
                
                if gt_idx >= all_frames.size(1):
                    break
                    
                gt_frame = all_frames[:, gt_idx]
                
                # Decode and Warp
                flow_0, flow_1, mask = decoder(target_lstm_state, skips, t_val)
                pred_frame = warping_module(frame_0, frame_1, flow_0, flow_1, mask)
                
                # Calculate Loss
                loss, _, _, _ = criterion(pred_frame, gt_frame, all_params)
                interval_loss += loss
                num_frames_predicted += 1

        # 5. Scaled Backpropagation
        if num_frames_predicted > 0:
            interval_avg_loss = interval_loss / num_frames_predicted
            
            # Scale the loss and call backward
            scaler.scale(interval_avg_loss).backward()
            
            # Unscale the gradients and update the optimizer
            scaler.step(optimizer)
            
            # Update the scaler for the next iteration
            scaler.update()
            
            running_loss += interval_avg_loss.item()
            
            print(f"   Train Video [{batch_idx+1}/{len(dataloader)}] | Interval: {interval} | Avg Loss: {interval_avg_loss.item():.4f}")

    return running_loss / len(dataloader)


def validate_one_epoch(encoder, lstm, decoder, warping_module, criterion, dataloader, device, subsample_factor=10):
    encoder.eval()
    lstm.eval()
    decoder.eval()
    
    running_loss = 0.0
    all_params = list(encoder.parameters()) + list(lstm.parameters()) + list(decoder.parameters())
    
    with torch.no_grad():
        for batch_idx, (context_frames, all_frames) in enumerate(dataloader):
            context_frames = context_frames.to(device)
            all_frames = all_frames.to(device)
            
            # --- Sequence Truncation to save VRAM and Time ---
            MAX_SEQ = 4
            full_seq_len = context_frames.size(1)
            
            if full_seq_len > MAX_SEQ:
                start_context_idx = random.randint(0, full_seq_len - MAX_SEQ)
                context_frames = context_frames[:, start_context_idx : start_context_idx + MAX_SEQ]
            else:
                start_context_idx = 0
                
            seq_len = context_frames.size(1)
            lstm_inputs = []
            skips_list = []
            
            for i in range(seq_len):
                features, f1, f2, f3 = encoder(context_frames[:, i])
                lstm_inputs.append(features)
                skips_list.append([f1, f2, f3])
                
            lstm_sequence = torch.stack(lstm_inputs, dim=1)
            lstm_outputs = lstm(lstm_sequence)
            
            # Evaluate on one random interval inside our chunk
            interval = random.randint(0, seq_len - 2)
            frame_0 = context_frames[:, interval]
            frame_1 = context_frames[:, interval + 1]
            target_lstm_state = lstm_outputs[:, interval]
            skips = skips_list[interval]
            
            # Calculate the absolute interval index for ground truth
            abs_interval = start_context_idx + interval
            
            interval_loss = 0.0
            num_frames_predicted = 0

            for offset in range(1, subsample_factor):
                t_val = offset / float(subsample_factor)
                gt_idx = (abs_interval * subsample_factor) + offset
                
                if gt_idx >= all_frames.size(1):
                    break
                    
                gt_frame = all_frames[:, gt_idx]
                flow_0, flow_1, mask = decoder(target_lstm_state, skips, t_val)
                pred_frame = warping_module(frame_0, frame_1, flow_0, flow_1, mask)
                
                loss, _, _, _ = criterion(pred_frame, gt_frame, all_params)
                interval_loss += loss
                num_frames_predicted += 1

            if num_frames_predicted > 0:
                running_loss += (interval_loss / num_frames_predicted).item()
                
    return running_loss / len(dataloader)
if __name__ == "__main__":
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_DIR = os.path.join(BASE_DIR, "KoNViD_1k_videos")
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    SUBSAMPLE_FACTOR = 2
    
    if torch.backends.mps.is_available():
        print("MPS backend is available. Training will utilize Apple Silicon GPU acceleration.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA backend is available. Training will utilize NVIDIA GPU acceleration.")
        device = torch.device("cuda")
    else:
        print("No GPU acceleration available. Training will run on CPU, which may be very slow.")
        device = torch.device("cpu")
    
    print(f"Executing on device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    # Grab all mp4 files and split them: 100 for training, 20 for validation
    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    if len(all_videos) < 120:
        print(f"Warning: Found {len(all_videos)} videos, which is less than 120.")
        train_videos = all_videos[:int(len(all_videos)*0.8)]
        val_videos = all_videos[int(len(all_videos)*0.8):]
    else:
        train_videos = all_videos[:100]
        val_videos = all_videos[100:120]

    train_dataset = FullVideoDataset(VIDEO_DIR, train_videos, SUBSAMPLE_FACTOR)
    val_dataset = FullVideoDataset(VIDEO_DIR, val_videos, SUBSAMPLE_FACTOR)

    # Batch size MUST be 1 since we are loading entire videos into RAM
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ==========================================
    # 3. Model & Optimizer Initialization
    # ==========================================
    encoder = CNNEncoder().to(device)
    lstm = BiConvLSTM(input_dim=256, hidden_dim=128).to(device)
    decoder = TimeConditionedDecoder().to(device)
    warping_module = WarpingModule().to(device)
    
    criterion = VFICombinedLoss(ssim_weight=0.1, charbonnier_weight=0.9, l1_weight=1e-4).to(device)
    
    all_params = list(encoder.parameters()) + list(lstm.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE)

    # ==========================================
    # 4. The Master Training Loop
    # ==========================================
    # ==========================================
    # 4. The Master Training Loop
    # ==========================================

    
    # --- NEW: Initialize the Mixed Precision Scaler for the selected device ---
    scaler = torch.amp.GradScaler(device.type)
    
    print("\n--- Starting Training Process ---")
    best_val_loss = float('inf')
    
    print("\n--- Starting Training Process ---")
    print("Press Ctrl+C at any time to safely pause and save your progress.")
    
    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
            
            # Train
            train_loss = train_one_epoch(
                encoder, lstm, decoder, warping_module, criterion,
                optimizer, train_loader, device, SUBSAMPLE_FACTOR
            )
            
            # Validate
            print("   Running Validation...")
            val_loss = validate_one_epoch(
                encoder, lstm, decoder, warping_module, criterion,
                val_loader, device, SUBSAMPLE_FACTOR
            )
            
            print(f"   => Epoch {epoch+1} Summary: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("   => Improved! Saving model weights...")
                torch.save(encoder.state_dict(), os.path.join(BASE_DIR, "best_encoder.pth"))
                torch.save(lstm.state_dict(), os.path.join(BASE_DIR, "best_lstm.pth"))
                torch.save(decoder.state_dict(), os.path.join(BASE_DIR, "best_decoder.pth"))

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user.")
        print("Saving current weights before exiting...")
        torch.save(encoder.state_dict(), os.path.join(BASE_DIR, "interrupted_encoder.pth"))
        torch.save(lstm.state_dict(), os.path.join(BASE_DIR, "interrupted_lstm.pth"))
        torch.save(decoder.state_dict(), os.path.join(BASE_DIR, "interrupted_decoder.pth"))
        print("Saved successfully! You can resume later.")

    print("\nTraining script terminated.")


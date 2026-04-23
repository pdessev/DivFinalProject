import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import your custom modules
from dataset import FullVideoDataset
from CNNv1 import CNNEncoder, BiConvLSTM, TimeConditionedDecoder
from warping import WarpingModule
from loss import VFICombinedLoss
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# Number of predicted frames to accumulate per backward pass.
# Lower = less memory, more LSTM backward passes (slower).
# Higher = more memory, fewer LSTM backward passes (faster).
BACKWARD_CHUNK_SIZE = 55


def train_one_epoch(encoder, lstm, decoder, warping_module, criterion, optimizer, dataloader, device, scaler=None, subsample_factor=10, raft_model=None):
    encoder.train()
    lstm.train()
    decoder.train()
    
    running_loss = 0.0
    all_params = list(encoder.parameters()) + list(lstm.parameters()) + list(decoder.parameters())

    RATE_REPORT_INTERVAL = 1.0  # seconds
    _rate_t0 = time.monotonic()
    _rate_cycles = 0
    _total_t0 = time.monotonic()
    _total_frames = 0

    for batch_idx, (context_frames, all_frames) in enumerate(dataloader):
        context_frames = context_frames.to(device)
        all_frames = all_frames.to(device)
        
        optimizer.zero_grad()
            
        seq_len = context_frames.size(1)
        lstm_inputs = []
        skips_list = []

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            # 1. Run the FULL sequence through the CNN Encoder
            for i in range(seq_len):
                features, f1, f2, f3 = encoder(context_frames[:, i])
                lstm_inputs.append(features)
                skips_list.append([f1, f2, f3])

            lstm_sequence = torch.stack(lstm_inputs, dim=1)

            # 2. Process the sequence through the BiConvLSTM
            lstm_outputs = lstm(lstm_sequence)

        # 3. Pre-compute the full list of frames to predict so we know the total
        #    count (for loss averaging) and can identify the last frame (to drop
        #    retain_graph and free the LSTM graph after the final backward).
        frames_to_predict = []
        for interval in range(seq_len - 1):
            for offset in range(1, subsample_factor):
                gt_idx = interval * subsample_factor + offset
                if gt_idx >= all_frames.size(1):
                    break
                frames_to_predict.append((interval, offset, offset / float(subsample_factor), gt_idx))

        n_frames = len(frames_to_predict)
        if n_frames == 0:
            continue

        # 4. Chunked backward: accumulate loss over BACKWARD_CHUNK_SIZE frames,
        #    then backward with retain_graph=True (keep LSTM graph alive between
        #    chunks). The final chunk uses retain_graph=False to release it.
        total_loss_item = 0.0
        for chunk_start in range(0, n_frames, BACKWARD_CHUNK_SIZE):
            chunk = frames_to_predict[chunk_start : chunk_start + BACKWARD_CHUNK_SIZE]
            is_last_chunk = (chunk_start + BACKWARD_CHUNK_SIZE >= n_frames)
            chunk_loss = 0.0

            for (interval, offset, t_val, gt_idx) in chunk:
                frame_0 = context_frames[:, interval]
                frame_1 = context_frames[:, interval + 1]
                gt_frame = all_frames[:, gt_idx]

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    gt_flow_0 = gt_flow_1 = None
                    if raft_model is not None:
                        with torch.no_grad():
                            gt_flow_0 = raft_model(frame_0 * 2.0 - 1.0, gt_frame * 2.0 - 1.0)[-1]
                            gt_flow_1 = raft_model(frame_1 * 2.0 - 1.0, gt_frame * 2.0 - 1.0)[-1]

                    lstm_state = (1.0 - t_val) * lstm_outputs[:, interval] + t_val * lstm_outputs[:, interval + 1]
                    skips = [
                        (1.0 - t_val) * skips_list[interval][s] + t_val * skips_list[interval + 1][s]
                        for s in range(3)
                    ]

                    flow_0, flow_1, mask = decoder(lstm_state, skips, t_val)

                    if flow_0.shape[2:] != frame_0.shape[2:]:
                        flow_0 = F.interpolate(flow_0, size=frame_0.shape[2:], mode='bilinear', align_corners=False)
                        flow_1 = F.interpolate(flow_1, size=frame_0.shape[2:], mode='bilinear', align_corners=False)
                        mask   = F.interpolate(mask,   size=frame_0.shape[2:], mode='bilinear', align_corners=False)

                    pred_frame, warped_0, warped_1 = warping_module(frame_0, frame_1, flow_0, flow_1, mask)

                    loss, _, _, _ = criterion(pred_frame, gt_frame, all_params,
                                              warped_0, warped_1,
                                              flow_0, flow_1,
                                              gt_flow_0, gt_flow_1)

                chunk_loss = chunk_loss + loss / n_frames
                total_loss_item += loss.item()

                _rate_cycles += 1
                _total_frames += 1
                _elapsed = time.monotonic() - _rate_t0
                if _elapsed >= RATE_REPORT_INTERVAL:
                    rate = _rate_cycles / _elapsed * 60.0
                    total_elapsed = time.monotonic() - _total_t0
                    avg_s_per_frame = total_elapsed / _total_frames
                    print(f"   Rate: {rate:.1f} frames/min  |  Total: {_total_frames} frames in {total_elapsed:.1f}s  ({avg_s_per_frame:.2f} s/frame avg)")
                    _rate_t0 = time.monotonic()
                    _rate_cycles = 0

            if scaler is not None:
                scaler.scale(chunk_loss).backward(retain_graph=not is_last_chunk)
            else:
                chunk_loss.backward(retain_graph=not is_last_chunk)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += total_loss_item / n_frames
        print(f"   Train Video [{batch_idx+1}/{len(dataloader)}] | Frames: {n_frames} | Avg Loss: {total_loss_item/n_frames:.4f}")

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
            
            seq_len = context_frames.size(1)
            lstm_inputs = []
            skips_list = []
            
            for i in range(seq_len):
                features, f1, f2, f3 = encoder(context_frames[:, i])
                lstm_inputs.append(features)
                skips_list.append([f1, f2, f3])
                
            lstm_sequence = torch.stack(lstm_inputs, dim=1)
            lstm_outputs = lstm(lstm_sequence)
            
            total_loss = 0.0
            total_frames_predicted = 0

            for interval in range(seq_len - 1):
                frame_0 = context_frames[:, interval]
                frame_1 = context_frames[:, interval + 1]

                for offset in range(1, subsample_factor):
                    t_val = offset / float(subsample_factor)
                    gt_idx = (interval * subsample_factor) + offset

                    if gt_idx >= all_frames.size(1):
                        break

                    gt_frame = all_frames[:, gt_idx]

                    lstm_state = (1.0 - t_val) * lstm_outputs[:, interval] + t_val * lstm_outputs[:, interval + 1]
                    skips = [
                        (1.0 - t_val) * skips_list[interval][s] + t_val * skips_list[interval + 1][s]
                        for s in range(3)
                    ]

                    flow_0, flow_1, mask = decoder(lstm_state, skips, t_val)

                    if flow_0.shape[2:] != frame_0.shape[2:]:
                        flow_0 = F.interpolate(flow_0, size=frame_0.shape[2:], mode='bilinear', align_corners=False)
                        flow_1 = F.interpolate(flow_1, size=frame_0.shape[2:], mode='bilinear', align_corners=False)
                        mask   = F.interpolate(mask,   size=frame_0.shape[2:], mode='bilinear', align_corners=False)

                    pred_frame, warped_0, warped_1 = warping_module(frame_0, frame_1, flow_0, flow_1, mask)

                    loss, _, _, _ = criterion(pred_frame, gt_frame, all_params,
                                              warped_0, warped_1,
                                              flow_0, flow_1)
                    total_loss += loss.item()
                    total_frames_predicted += 1

            if total_frames_predicted > 0:
                running_loss += total_loss / total_frames_predicted
                
    return running_loss / len(dataloader)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_DIR = os.path.join(BASE_DIR, "KoNViD_1k_videos")
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    SUBSAMPLE_FACTOR = 4
    TRAINING_SET_SIZE = 80
    RESOLUTION = (512, 288)  # (width, height)
    
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

    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    if len(all_videos) < TRAINING_SET_SIZE:
        train_videos = all_videos[:int(len(all_videos)*0.8)]
        val_videos = all_videos[int(len(all_videos)*0.8):]
    else:
        train_videos = all_videos[:TRAINING_SET_SIZE]
        val_videos = all_videos[TRAINING_SET_SIZE:TRAINING_SET_SIZE + (TRAINING_SET_SIZE // 5)]

    train_dataset = FullVideoDataset(VIDEO_DIR, train_videos, SUBSAMPLE_FACTOR, resize_dims=RESOLUTION)
    val_dataset = FullVideoDataset(VIDEO_DIR, val_videos, SUBSAMPLE_FACTOR, resize_dims=RESOLUTION)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    encoder = CNNEncoder().to(device)
    lstm = BiConvLSTM(input_dim=256, hidden_dim=128).to(device)
    decoder = TimeConditionedDecoder().to(device)
    warping_module = WarpingModule().to(device)
    
    criterion = VFICombinedLoss(
        ssim_weight=0.1, charbonnier_weight=0.1, gradient_weight=0.1,
        warp_weight=0.6, l1_weight=1e-4, perceptual_weight=0.05,
        flow_supervision_weight=0.01, flow_consistency_weight=0.01,
        flow_tv_weight=0.005
    ).to(device)

    # Frozen RAFT model used as a pseudo-GT flow label generator during training.
    # Weights are downloaded automatically on first run (~20 MB).
    raft_weights = Raft_Small_Weights.DEFAULT
    raft_model = raft_small(weights=raft_weights).to(device)
    raft_model.eval()
    for param in raft_model.parameters():
        param.requires_grad = False
    print("RAFT optical flow model loaded.")

    all_params = list(encoder.parameters()) + list(lstm.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')

    print("\n--- Starting Training Process ---")

    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

            train_loss = train_one_epoch(
                encoder, lstm, decoder, warping_module, criterion,
                optimizer, train_loader, device, scaler, SUBSAMPLE_FACTOR,
                raft_model=raft_model
            )
            
            print("   Running Validation...")
            val_loss = validate_one_epoch(
                encoder, lstm, decoder, warping_module, criterion,
                val_loader, device, SUBSAMPLE_FACTOR
            )
            
            print(f"   => Epoch {epoch+1}: Train = {train_loss:.4f} | Val = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(encoder.state_dict(), os.path.join(BASE_DIR, "best_encoder.pth"))
                torch.save(lstm.state_dict(), os.path.join(BASE_DIR, "best_lstm.pth"))
                torch.save(decoder.state_dict(), os.path.join(BASE_DIR, "best_decoder.pth"))

    except KeyboardInterrupt:
        torch.save(encoder.state_dict(), os.path.join(BASE_DIR, "interrupted_encoder.pth"))
        torch.save(lstm.state_dict(), os.path.join(BASE_DIR, "interrupted_lstm.pth"))
        torch.save(decoder.state_dict(), os.path.join(BASE_DIR, "interrupted_decoder.pth"))
        print("\nTraining interrupted and weights saved.")

    print("\nTraining complete.")

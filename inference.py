import os
import cv2
import torch
import numpy as np

from dataset import FullVideoDataset
from CNNv1 import CNNEncoder, BiConvLSTM, TimeConditionedDecoder
from warping import WarpingModule

USE_INTERRUPT = True  # Set to False to load the "best" models instead of "interrupt" models

# -----------------------------
# Model Loading
# -----------------------------
def load_models(device, encoder_path, lstm_path, decoder_path):
    encoder = CNNEncoder().to(device)
    lstm = BiConvLSTM(input_dim=256, hidden_dim=128).to(device)
    decoder = TimeConditionedDecoder().to(device)
    warping = WarpingModule().to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    lstm.eval()
    decoder.eval()

    return encoder, lstm, decoder, warping


# -----------------------------
# Tensor → OpenCV frame
# -----------------------------
def tensor_to_frame(t):
    img = t.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# -----------------------------
# Full-video inference
# -----------------------------
def run_inference_on_video(
    video_dir,
    video_name,
    subsample_factor,
    device,
    encoder,
    lstm,
    decoder,
    warping,
    output_path="comparison.mp4"
):

    dataset = FullVideoDataset(video_dir, [video_name], subsample_factor)
    context_frames, all_frames, file_names = dataset[0]

    context_frames = context_frames.unsqueeze(0).to(device)  # [1,T,C,H,W]
    all_frames = all_frames.unsqueeze(0).to(device)

    B, T, C, H, W = context_frames.shape

    # -----------------------------
    # Encode ALL context frames
    # -----------------------------
    features_list = []
    skips_list = []

    with torch.no_grad():
        for t in range(T):
            feat, f1, f2, f3 = encoder(context_frames[:, t])
            features_list.append(feat)
            skips_list.append([f1, f2, f3])

        features_seq = torch.stack(features_list, dim=1)
        lstm_out = lstm(features_seq)

    # -----------------------------
    # Generate FULL video timeline
    # -----------------------------
    # original_frames: left side — the original video (context + ground-truth intermediates)
    # supersampled_frames: right side — context frames + model-predicted intermediates
    original_frames = []
    supersampled_frames = []

    with torch.no_grad():

        for interval in range(T - 1):

            frame_0 = context_frames[:, interval]
            frame_1 = context_frames[:, interval + 1]

            # Include the context frame on both sides
            gt_context_idx = interval * subsample_factor
            original_frames.append(all_frames[:, gt_context_idx])
            supersampled_frames.append(frame_0)

            # Generate and collect intermediate frames
            for offset in range(1, subsample_factor):

                t_val = offset / float(subsample_factor)
                gt_idx = interval * subsample_factor + offset

                if gt_idx >= all_frames.shape[1]:
                    continue

                gt_frame = all_frames[:, gt_idx]

                # t-weighted interpolation gives the decoder symmetric context:
                # at t=0.5 it sees equal contributions from both bounding frames.
                lstm_state = (1.0 - t_val) * lstm_out[:, interval] + t_val * lstm_out[:, interval + 1]
                skips = [
                    (1.0 - t_val) * skips_list[interval][s] + t_val * skips_list[interval + 1][s]
                    for s in range(3)
                ]

                flow_0, flow_1, mask = decoder(lstm_state, skips, t_val)

                # resize safety
                if flow_0.shape[2:] != frame_0.shape[2:]:
                    flow_0 = torch.nn.functional.interpolate(
                        flow_0, size=frame_0.shape[2:], mode='bilinear', align_corners=False
                    )
                    flow_1 = torch.nn.functional.interpolate(
                        flow_1, size=frame_0.shape[2:], mode='bilinear', align_corners=False
                    )
                    mask = torch.nn.functional.interpolate(
                        mask, size=frame_0.shape[2:], mode='bilinear', align_corners=False
                    )

                pred_frame, _, _ = warping(frame_0, frame_1, flow_0, flow_1, mask)

                original_frames.append(gt_frame)
                supersampled_frames.append(pred_frame)

        # Include the final context frame on both sides
        final_gt_idx = (T - 1) * subsample_factor
        if final_gt_idx < all_frames.shape[1]:
            original_frames.append(all_frames[:, final_gt_idx])
        supersampled_frames.append(context_frames[:, T - 1])


    # -----------------------------
    # Write video
    # -----------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sample = tensor_to_frame(original_frames[0][0])
    H, W, _ = sample.shape

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (W * 2, H)
    )

    for orig, sup in zip(original_frames, supersampled_frames):
        orig_img = tensor_to_frame(orig[0])
        sup_img  = tensor_to_frame(sup[0])

        combined = np.concatenate([orig_img, sup_img], axis=1)
        writer.write(combined)

    writer.release()

    print(f"\nSaved comparison video → {output_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if torch.backends.mps.is_available():
        print("MPS backend is available. Inferencing will utilize Apple Silicon GPU acceleration.")
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA backend is available. Inferencing will utilize NVIDIA GPU acceleration.")
        DEVICE = torch.device("cuda")
    else:
        print("No GPU acceleration available. Inferencing will run on CPU, which may be very slow.")
        DEVICE = torch.device("cpu")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    encoder_prefix = "interrupted" if USE_INTERRUPT else "best"
    
    VIDEO_DIR = os.path.join(BASE_DIR, "KoNViD_1k_videos")
    ENCODER_PATH = os.path.join(BASE_DIR, f"{encoder_prefix}_encoder.pth")
    LSTM_PATH = os.path.join(BASE_DIR, f"{encoder_prefix}_lstm.pth")
    DECODER_PATH = os.path.join(BASE_DIR, f"{encoder_prefix}_decoder.pth")
    VIDEO_NAME = "9908050493.mp4"

    SUBSAMPLE_FACTOR = 2

    encoder, lstm, decoder, warping = load_models(
        DEVICE,
        ENCODER_PATH,
        LSTM_PATH,
        DECODER_PATH
    )

    run_inference_on_video(
        VIDEO_DIR,
        VIDEO_NAME,
        SUBSAMPLE_FACTOR,
        DEVICE,
        encoder,
        lstm,
        decoder,
        warping,
        output_path=os.path.join(BASE_DIR, "comparison.mp4")
    )

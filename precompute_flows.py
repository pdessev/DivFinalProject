"""
Precompute RAFT optical flows for all videos before training.

Run once:
    python precompute_flows.py

Flows are written to  flows_cache/<video_stem>.pt  (dict keyed by
(interval, offset) tuples, each value is a pair of CPU tensors [2, H, W]).
Training will pick them up automatically and skip live RAFT inference.

RESOLUTION and SUBSAMPLE_FACTOR must match the values in train.py.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# ── Must match train.py ───────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR        = os.path.join(BASE_DIR, "KoNViD_1k_videos")
CACHE_DIR        = os.path.join(BASE_DIR, "flows_cache")
RESOLUTION       = (512, 288)   # (width, height)
SUBSAMPLE_FACTOR = 4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

weights = Raft_Small_Weights.DEFAULT
raft = raft_small(weights=weights).to(device)
raft.eval()
for p in raft.parameters():
    p.requires_grad = False


def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, RESOLUTION, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    cap.release()
    return frames


videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
print(f"Found {len(videos)} videos.\n")

for vid_idx, video_name in enumerate(videos):
    stem       = Path(video_name).stem
    cache_path = os.path.join(CACHE_DIR, f"{stem}.pt")

    if os.path.exists(cache_path):
        print(f"[{vid_idx+1}/{len(videos)}] {video_name}: already cached, skipping.")
        continue

    print(f"[{vid_idx+1}/{len(videos)}] {video_name}: loading frames…")
    raw_frames = load_video_frames(os.path.join(VIDEO_DIR, video_name))

    if not raw_frames:
        print("  WARNING: no frames, skipping.")
        continue

    all_frames     = torch.stack(raw_frames)             # [T, 3, H, W]
    context_frames = all_frames[::SUBSAMPLE_FACTOR]      # [S, 3, H, W]
    seq_len        = context_frames.size(0)

    flows = {}
    total = sum(
        1
        for interval in range(seq_len - 1)
        for offset in range(1, SUBSAMPLE_FACTOR)
        if interval * SUBSAMPLE_FACTOR + offset < all_frames.size(0)
    )
    done = 0

    with torch.no_grad():
        for interval in range(seq_len - 1):
            f0 = context_frames[interval    ].unsqueeze(0).to(device)  # [1,3,H,W]
            f1 = context_frames[interval + 1].unsqueeze(0).to(device)

            for offset in range(1, SUBSAMPLE_FACTOR):
                gt_idx = interval * SUBSAMPLE_FACTOR + offset
                if gt_idx >= all_frames.size(0):
                    break

                gt = all_frames[gt_idx].unsqueeze(0).to(device)        # [1,3,H,W]

                # RAFT expects pixel values in [-1, 1]
                flow_0 = raft(f0 * 2.0 - 1.0, gt * 2.0 - 1.0)[-1]    # [1,2,H,W]
                flow_1 = raft(f1 * 2.0 - 1.0, gt * 2.0 - 1.0)[-1]

                flows[(interval, offset)] = (
                    flow_0.squeeze(0).cpu(),   # [2,H,W] — no batch dim to save memory
                    flow_1.squeeze(0).cpu(),
                )

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} flows computed…")

    torch.save(flows, cache_path)
    print(f"  Saved {len(flows)} flow pairs → {cache_path}\n")

print("Precomputation complete.")

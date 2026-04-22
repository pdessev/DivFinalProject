
import os
import glob
import cv2
import torch

# ==========================================
# 1. Hardware Configuration
# ==========================================
# Configurable for Apple Silicon (M2/MPS) or NVIDIA (CUDA)
if torch.backends.mps.is_available():
    # Apple Silicon hardware acceleration (M1/M2/M3)
    device = torch.device("mps")
    print("Hardware: Using Apple Silicon (M2/MPS).")
elif torch.cuda.is_available():
    # NVIDIA GPU hardware acceleration
    device = torch.device("cuda")
    print("Hardware: Using NVIDIA GPU (CUDA).")
else:
    # Fallback to CPU
    device = torch.device("cpu")
    print("Hardware: Using CPU. No acceleration found.")

# ==========================================
# 2. Directory Setup
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_dir, "KoNViD_1k_videos")
output_file = os.path.join(base_dir, "subsampletest1.mp4")

# ==========================================
# 3. Video Subsampling & Visualization
# ==========================================
def create_subsampled_comparison():
    # Grab the first .mp4 file we can find in the directory
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {video_dir}. Please check the path.")
    
    input_video_path = video_files[0]
    print(f"Selected video for subsampling: {os.path.basename(input_video_path)}")

    # Open the video reader
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {input_video_path}")

    # Extract original video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # We are creating a side-by-side video, so the output width is doubled
    out_width = width * 2
    out_height = height

    # Define the codec and create VideoWriter object ('mp4v' works well on Mac)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (out_width, out_height))

    subsample_factor = 10
    held_frame = None

    print(f"Original FPS: {fps:.2f}. Processing {total_frames} frames...")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update the 'held_frame' only on every 10th frame
        # This simulates extracting features only once every 10 frames,
        # while keeping it on screen for 10x the duration.
        if i % subsample_factor == 0:
            held_frame = frame.copy()
            
        # Safety fallback for the very first iteration
        if held_frame is None:
            held_frame = frame.copy()

        # Stitch them together: Original on the left, Subsampled on the right
        side_by_side = cv2.hconcat([frame, held_frame])
        
        # Write the combined frame to our output video
        out.write(side_by_side)

    # Clean up the resources
    cap.release()
    out.release()
    print(f"Success! Side-by-side comparison saved to: {output_file}")

if __name__ == "__main__":
    create_subsampled_comparison() 

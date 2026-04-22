import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FullVideoDataset(Dataset):
    """
    Reads an entire video into memory.
    Yields the subsampled context frames for the LSTM, and the complete set of frames for ground truth.
    """
    def __init__(self, video_dir, video_filenames, subsample_factor=10):
        """
        Args:
            video_dir (str): Path to the folder containing the MP4s.
            video_filenames (list): List of video file names (e.g., for train or test split).
            subsample_factor (int): The gap between known context frames. Default is 10.
        """
        self.video_dir = video_dir
        self.video_filenames = video_filenames
        self.subsample_factor = subsample_factor

    def __len__(self):
        # One epoch means the network has processed every video in this specific list once
        return len(self.video_filenames)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_filenames[idx])
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video reached
            
            # OpenCV loads videos in BGR format; PyTorch expects RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert pixel values from [0, 255] integer range to [0.0, 1.0] float range
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to tensor and permute from [Height, Width, Channels] to [Channels, Height, Width]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame_tensor)
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Video file {video_path} contained no frames.")
        
        # Stack all individual frame tensors into a single massive tensor
        # Shape: [Total_Frames, 3, Height, Width] (e.g., [240, 3, 540, 960])
        all_frames = torch.stack(frames)
        
        # Slice out just the known context frames using Python's step slicing
        # Shape: [Context_Length, 3, Height, Width] (e.g., [24, 3, 540, 960])
        context_frames = all_frames[::self.subsample_factor]
        
        return context_frames, all_frames


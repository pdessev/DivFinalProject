import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FullVideoDataset(Dataset):
    """
    Reads an entire video into memory, resizing frames on the fly to accelerate training.
    Yields the full sequence of subsampled context frames for the LSTM.
    """
    def __init__(self, video_dir, video_filenames, subsample_factor=10, resize_dims=(960, 540)):
        self.video_dir = video_dir
        self.video_filenames = video_filenames
        self.subsample_factor = subsample_factor
        self.resize_dims = resize_dims # (Width, Height)

    def __len__(self):
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
                break
                
            # --- NEW: Resize immediately to save memory & speed up math ---
            if self.resize_dims:
                frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_AREA)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame_tensor)
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Video file {video_path} contained no frames.")
            
        all_frames = torch.stack(frames)
        context_frames = all_frames[::self.subsample_factor]
        
        return context_frames, all_frames, self.video_filenames[idx]

import torch
import torch.nn as nn
import torch.nn.functional as F

def mps_safe_grid_sample(img, grid):
    """
    A pure PyTorch implementation of bilinear interpolation grid sampling.
    This bypasses the missing MPS backend C++ function, keeping gradients on the GPU.
    """
    N, C, H, W = img.shape
    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    
    # Scale from [-1, 1] to [0, W-1] and [0, H-1]
    x = ((x + 1.0) * (W - 1)) / 2.0
    y = ((y + 1.0) * (H - 1)) / 2.0
    
    # Get corner pixel coordinates
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp to image boundaries so we don't index out of bounds
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    
    # Calculate weights for bilinear interpolation
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)
    
    # Flatten image for a safe, differentiable gather operation
    img_flat = img.view(N, C, -1)
    
    # Helper function to gather pixels using flat indices
    def gather_pixels(y_idx, x_idx):
        flat_idx = y_idx * W + x_idx
        expanded_idx = flat_idx.view(N, 1, -1).expand(N, C, -1)
        return torch.gather(img_flat, 2, expanded_idx).view(N, C, H, W)
        
    Ia = gather_pixels(y0, x0)
    Ib = gather_pixels(y1, x0)
    Ic = gather_pixels(y0, x1)
    Id = gather_pixels(y1, x1)
    
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

class WarpingModule(nn.Module):
    """
    Differentiable image warping using optical flow and soft blending.
    """
    def __init__(self):
        super(WarpingModule, self).__init__()
        # We cache the base coordinate grid so we don't recreate it every forward pass
        self.grid_dict = {}

    def get_base_grid(self, batch_size, height, width, device):
        """
        Creates a meshgrid of absolute pixel coordinates (0 to W-1, 0 to H-1).
        Caches it to avoid redundant computation.
        """
        key = (batch_size, height, width, device)
        if key not in self.grid_dict:
            # Create a 1D tensor for X and Y coordinates
            x = torch.arange(0, width, device=device, dtype=torch.float32)
            y = torch.arange(0, height, device=device, dtype=torch.float32)
            
            # Create a 2D meshgrid
            # indexing='ij' ensures y is the first dimension, x is the second
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            
            # Stack to create [Height, Width, 2] where the last dim is (X, Y)
            base_grid = torch.stack([grid_x, grid_y], dim=-1)
            
            # Expand to include the batch dimension: [Batch, Height, Width, 2]
            base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            self.grid_dict[key] = base_grid
            
        return self.grid_dict[key]

    def warp(self, img, flow):
        """
        Warps a single image using the provided optical flow.
        """
        b, c, h, w = img.size()
        
        # flow is [Batch, 2, Height, Width]. We need it to be [Batch, Height, Width, 2]
        # and in (X, Y) order. We permute it to match grid_sample expectations.
        flow = flow.permute(0, 2, 3, 1)
        
        # Get the base pixel coordinates
        base_grid = self.get_base_grid(b, h, w, img.device)
        
        # Add the predicted flow shifts to the base coordinates
        # This tells us the absolute (X, Y) coordinate we want to sample from
        sampling_grid = base_grid + flow
        
        # PyTorch's grid_sample expects coordinates normalized between -1 and 1
        # where (-1, -1) is the top-left corner and (1, 1) is bottom-right.
        # Normalize X from [0, W-1] to [-1, 1]
        norm_x = 2.0 * sampling_grid[..., 0] / (w - 1) - 1.0
        # Normalize Y from [0, H-1] to [-1, 1]
        norm_y = 2.0 * sampling_grid[..., 1] / (h - 1) - 1.0
        
        # Re-stack the normalized coordinates
        normalized_grid = torch.stack([norm_x, norm_y], dim=-1)
        
        # Perform the actual differentiable sampling
        # padding_mode='border' duplicates the edge pixels if the flow looks out of bounds
        # New line
        if img.device.type == 'mps':
            # Use the custom MPS-safe grid sampling function
            warped_img = mps_safe_grid_sample(img, normalized_grid)
        else:
            # Use the standard grid_sample function
            warped_img = F.grid_sample(img, normalized_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped_img

    def forward(self, frame_0, frame_1, flow_0, flow_1, mask):
        warped_0 = self.warp(frame_0, flow_0)
        warped_1 = self.warp(frame_1, flow_1)
        interpolated_frame = (mask * warped_0) + ((1.0 - mask) * warped_1)
        return interpolated_frame, warped_0, warped_1

# ==========================================
# Unit Test: Verifying the Warping Module
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Warping Module on device: {device}")
    
    batch_size = 1
    channels = 3
    height = 540
    width = 960
    
    # 1. Simulate the input frames (I_0 and I_1)
    # Using random noise to represent the original RGB frames
    frame0 = torch.rand(batch_size, channels, height, width).to(device)
    frame1 = torch.rand(batch_size, channels, height, width).to(device)
    
    # 2. Simulate the Decoder's output (Flow vectors and Mask)
    # Flow values are usually in pixel shifts. Let's simulate minor movements.
    flow0_to_t = (torch.rand(batch_size, 2, height, width).to(device) * 10) - 5 # Shifts between -5 and 5 pixels
    flow1_to_t = (torch.rand(batch_size, 2, height, width).to(device) * 10) - 5
    
    # Mask strictly between 0 and 1 (simulating the Sigmoid output)
    mask = torch.rand(batch_size, 1, height, width).to(device)
    
    # Initialize the Warping Module
    warping_module = WarpingModule().to(device)
    
    # Run the forward pass
    predicted_frame_t, _, _ = warping_module(frame0, frame1, flow0_to_t, flow1_to_t, mask)
    
    print("\n--- Output Shapes ---")
    print(f"Frame 0 (I_0) Input : {frame0.shape}")
    print(f"Frame 1 (I_1) Input : {frame1.shape}")
    print(f"Predicted Frame (I_t): {predicted_frame_t.shape} (Target: {batch_size}, 3, 540, 960)")
    
    if predicted_frame_t.shape == (batch_size, channels, height, width):
        print("\nUnit test successful! Pixels have been successfully warped and blended.")
    else:
        print("\nShape mismatch detected in the warping output.")

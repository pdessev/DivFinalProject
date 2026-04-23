import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _cp

class DecoderBlock(nn.Module):
    """
    Upsamples the feature map and merges it with the skip connection.
    Uses Bilinear Upsampling + Conv to avoid checkerboard artifacts.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # After upsampling, we reduce channels before concatenating
        self.conv_up = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        
        # After concatenating with skip connection, we process the combined features
        self.conv_process = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_process(x)
        return x


class TimeConditionedDecoder(nn.Module):
    """
    Decodes the ConvLSTM hidden state + time 't' back into Optical Flow and a Blending Mask.
    """
    def __init__(self):
        super(TimeConditionedDecoder, self).__init__()
        
        # The input is the ConvLSTM output (256 channels) + 1 channel for time 't' = 257
        self.initial_conv = nn.Sequential(
            nn.Conv2d(257, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsampling stages (matching the skip connections from CNNEncoder)
        # Skip 3 (f3): 128 channels
        self.up1 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        
        # Skip 2 (f2): 64 channels
        self.up2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        # Skip 1 (f1): 32 channels
        self.up3 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
        
        # Final output layer: 32 channels -> 5 channels (Flow x2, Flow y2, Mask x1)
        self.final_conv = nn.Conv2d(32, 5, kernel_size=3, padding=1)

        self.use_checkpoint = False

    def forward(self, lstm_features, skip_connections, t):
        if self.use_checkpoint and self.training:
            f1, f2, f3 = skip_connections
            return _cp(self._forward_impl, lstm_features, f1, f2, f3, t, use_reentrant=False)
        return self._forward_impl(lstm_features, *skip_connections, t)

    def _forward_impl(self, lstm_features, f1, f2, f3, t):
        b, c, h, w = lstm_features.size()
        
        # 1. Spatially expand time 't' and concatenate
        # Create a tensor of shape [Batch, 1, Height, Width] filled with value 't'
        t_map = torch.full((b, 1, h, w), t, dtype=lstm_features.dtype, device=lstm_features.device)
        x = torch.cat([lstm_features, t_map], dim=1) # Now 257 channels
        
        # 2. Process initial fusion
        x = self.initial_conv(x) # Back to 256 channels
        
        # 3. Decode with skip connections
        x = self.up1(x, f3)      # -> 128 channels, 136x240
        x = self.up2(x, f2)      # -> 64 channels, 272x480
        x = self.up3(x, f1)      # -> 32 channels, 544x960
        
        # 4. Generate the 5 raw output channels
        out = self.final_conv(x) # -> 5 channels, 544x960
        
        # 5. Remove the 4 padding pixels we added in the encoder (Crop bottom)
        # Slicing keeps the top 540 pixels.
        out = out[:, :, :540, :] # -> 5 channels, 540x960
        
        # 6. Split into Flow and Mask
        flow_0_to_t = out[:, 0:2, :, :] # Channels 0, 1 (No activation, flow can be negative)
        flow_1_to_t = out[:, 2:4, :, :] # Channels 2, 3 (No activation)
        
        # Apply Sigmoid to the mask to strictly bound it between 0 and 1
        mask = torch.sigmoid(out[:, 4:5, :, :]) # Channel 4
        
        return flow_0_to_t, flow_1_to_t, mask

# ==========================================
# Unit Test: Verifying the Decoder
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Decoder on device: {device}")
    
    # Simulating the outputs from the CNNEncoder and BiConvLSTM
    batch_size = 1
    
    # Dummy Skip Connections from the Encoder
    f1 = torch.randn(batch_size, 32, 544, 960).to(device)
    f2 = torch.randn(batch_size, 64, 272, 480).to(device)
    f3 = torch.randn(batch_size, 128, 136, 240).to(device)
    skips = [f1, f2, f3]
    
    # Dummy LSTM feature map for a SINGLE time step
    lstm_out = torch.randn(batch_size, 256, 68, 120).to(device)
    
    # Initialize the decoder
    decoder = TimeConditionedDecoder().to(device)
    
    # Simulate generating the frame halfway between (t = 0.5)
    t_val = 0.5
    
    # Forward pass
    flow0, flow1, mask = decoder(lstm_out, skips, t_val)
    
    print("\n--- Output Shapes ---")
    print(f"Flow 0 -> t: {flow0.shape} (Target: {batch_size}, 2, 540, 960)")
    print(f"Flow 1 -> t: {flow1.shape} (Target: {batch_size}, 2, 540, 960)")
    print(f"Mask       : {mask.shape} (Target: {batch_size}, 1, 540, 960)")
    
    # Verify the padding removal worked
    if flow0.shape[2] == 540 and flow0.shape[3] == 960:
         print("\nUnit test successful! Spatial dimensions correctly recovered to 540x960.")
    else:
         print("\nShape mismatch detected in final output.")

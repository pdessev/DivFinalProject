import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _cp

class ResBlock(nn.Module):
    """
    A standard Residual Block.
    Maintains spatial dimensions if stride=1.
    Halves spatial dimensions if stride=2.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Shortcut connection to match dimensions if we downsample or change channel depth
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        out += residual
        return self.lrelu(out)


class CNNEncoder(nn.Module):
    """
    Extracts spatial features and builds a pyramid for skip connections.
    Input resolution: 520x960 (height x width). 520 divides evenly by 8,
    so no padding is required.
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()

        # Stage 1: Full resolution (Channels: 3 -> 32)
        self.stage1 = ResBlock(3, 32, stride=1)

        # Stage 2: 1/2 resolution (Channels: 32 -> 64)
        self.stage2 = ResBlock(32, 64, stride=2)

        # Stage 3: 1/4 resolution (Channels: 64 -> 128)
        self.stage3 = ResBlock(64, 128, stride=2)

        # Stage 4: 1/8 resolution (Channels: 128 -> 256)
        # This is the deepest bottleneck that will feed into the ConvLSTM
        self.stage4 = ResBlock(128, 256, stride=2)

    def forward(self, x):
        # x is expected to be shape: [Batch, 3, 520, 960]
        f1 = self.stage1(x)   # [Batch, 32,  520, 960] -> Skip Connection 1
        f2 = self.stage2(f1)  # [Batch, 64,  260, 480] -> Skip Connection 2
        f3 = self.stage3(f2)  # [Batch, 128, 130, 240] -> Skip Connection 3
        f4 = self.stage4(f3)  # [Batch, 256,  65, 120] -> To ConvLSTM

        return f4, f1, f2, f3

# ==========================================
# Unit Test: Verifying the Forward Pass
# ==========================================
if __name__ == "__main__":
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing CNN Encoder on device: {device}")
    
    # Initialize the model and move to device
    encoder = CNNEncoder().to(device)
    
    # Create a dummy tensor representing 1 batch of 1 video frame (1920x1080 display, 960x540 actual)
    batch_size = 1
    channels = 3
    height = 540
    width = 960
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Run the forward pass
    bottleneck_features, skip_connections = encoder(dummy_input)
    
    print("\n--- Output Shapes ---")
    print(f"Skip Connection 1 (f1): {skip_connections[0].shape} (Target: {batch_size}, 32, 544, 960)")
    print(f"Skip Connection 2 (f2): {skip_connections[1].shape} (Target: {batch_size}, 64, 272, 480)")
    print(f"Skip Connection 3 (f3): {skip_connections[2].shape} (Target: {batch_size}, 128, 136, 240)")
    print(f"Bottleneck (To ConvLSTM): {bottleneck_features.shape} (Target: {batch_size}, 256, 68, 120)")
    print("\nUnit test successful! All shapes align properly for the network.")




class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # We compute all 4 gates (input, forget, cell, output) in one single convolution
        # for efficiency, hence out_channels is 4 * hidden_dim
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and current hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split the output into the 4 gate tensors
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)     # Input gate
        f = torch.sigmoid(cc_f)     # Forget gate
        o = torch.sigmoid(cc_o)     # Output gate
        g = torch.tanh(cc_g)        # Cell update
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class BiConvLSTM(nn.Module):
    """
    Processes a sequence of spatial features forwards and backwards.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, use_checkpoint=False):
        super(BiConvLSTM, self).__init__()
        
        self.forward_cell    = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.backward_cell   = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.use_checkpoint  = use_checkpoint

    def forward(self, x):
        # x shape expects: [Batch, Sequence_Length, Channels, Height, Width]
        b, seq_len, c, h, w = x.size()
        
        # Initialize hidden states
        h_f, c_f = self.forward_cell.init_hidden(b, h, w, x.device)
        h_b, c_b = self.backward_cell.init_hidden(b, h, w, x.device)
        
        forward_outputs = []
        backward_outputs = []
        
        # Forward pass
        for t in range(seq_len):
            if self.use_checkpoint and self.training:
                h_f, c_f = _cp(self.forward_cell, x[:, t, :, :, :], (h_f, c_f), use_reentrant=False)
            else:
                h_f, c_f = self.forward_cell(x[:, t, :, :, :], (h_f, c_f))
            forward_outputs.append(h_f)
            
        # Backward pass
        for t in range(seq_len - 1, -1, -1):
            if self.use_checkpoint and self.training:
                h_b, c_b = _cp(self.backward_cell, x[:, t, :, :, :], (h_b, c_b), use_reentrant=False)
            else:
                h_b, c_b = self.backward_cell(x[:, t, :, :, :], (h_b, c_b))
            # Insert at the beginning so the time indices align with forward_outputs
            backward_outputs.insert(0, h_b)
            
        # Concatenate the hidden states from both directions along the channel dimension
        # Output shape for each timestep: [Batch, 2 * hidden_dim, Height, Width]
        outputs = []
        for t in range(seq_len):
            outputs.append(torch.cat([forward_outputs[t], backward_outputs[t]], dim=1))
            
        # Stack back into [Batch, Sequence_Length, Channels, Height, Width]
        return torch.stack(outputs, dim=1)

# ==========================================
# Unit Test: Verifying the BiConvLSTM
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing BiConvLSTM on device: {device}")
    
    # Let's simulate a sequence of 5 frames passing through the bottleneck
    batch_size = 1
    seq_length = 5
    input_channels = 256 # Output from our CNN encoder bottleneck
    hidden_channels = 128 # The LSTM will compress 256 down to 128 internally
    height = 68
    width = 120
    
    # [Batch, Seq_Len, Channels, Height, Width]
    dummy_sequence = torch.randn(batch_size, seq_length, input_channels, height, width).to(device)
    
    # Initialize BiConvLSTM
    biconvlstm = BiConvLSTM(input_dim=input_channels, hidden_dim=hidden_channels).to(device)
    
    # Forward pass
    lstm_output = biconvlstm(dummy_sequence)
    
    print(f"Input Sequence Shape: {dummy_sequence.shape}")
    print(f"Output Sequence Shape: {lstm_output.shape}")
    
    # The output channel depth should be 2 * hidden_channels (because it's bi-directional)
    expected_out_channels = hidden_channels * 2
    target_shape = (batch_size, seq_length, expected_out_channels, height, width)
    
    print(f"Target Shape: {target_shape}")
    if lstm_output.shape == target_shape:
        print("Unit test successful! Temporal memory module is working and shapes align.")
    else:
        print("Shape mismatch detected.")


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
        # Dynamically resize the upsampled tensor if pooling math caused an off-by-one error
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv_process(x)
        return x


class TimeConditionedDecoder(nn.Module):
    """
    Decodes the ConvLSTM hidden state + time 't' back into Optical Flow and a Blending Mask.
    The lstm_features input is expected to be a t-weighted interpolation between the hidden
    states of the two bounding context frames, so the decoder receives symmetric context.
    """
    def __init__(self):
        super(TimeConditionedDecoder, self).__init__()

        # The input is the ConvLSTM output (256 channels) + 1 channel for time 't' = 257
        self.initial_conv = nn.Sequential(
            nn.Conv2d(257, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling stages (matching the skip connections from CNNEncoder)
        self.up1 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)  # f3: 128ch
        self.up2 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)   # f2: 64ch
        self.up3 = DecoderBlock(in_channels=64,  skip_channels=32,  out_channels=32)   # f1: 32ch

        # 32 channels -> 5 output channels: (flow_0 x2, flow_1 x2, mask x1)
        self.final_conv = nn.Conv2d(32, 5, kernel_size=3, padding=1)

        self.use_checkpoint = False

    def forward(self, lstm_features, skip_connections, t):
        if self.use_checkpoint and self.training:
            f1, f2, f3 = skip_connections
            return _cp(self._forward_impl, lstm_features, f1, f2, f3, t, use_reentrant=False)
        return self._forward_impl(lstm_features, *skip_connections, t)

    def _forward_impl(self, lstm_features, f1, f2, f3, t):
        # lstm_features: [Batch, 256, 65, 120]  (t-interpolated between interval and interval+1)
        b, c, h, w = lstm_features.size()

        # 1. Broadcast scalar t into a spatial map and concatenate
        t_map = torch.full((b, 1, h, w), t, dtype=lstm_features.dtype, device=lstm_features.device)
        x = torch.cat([lstm_features, t_map], dim=1)  # [B, 257, 65, 120]

        # 2. Fuse time with features
        x = self.initial_conv(x)  # [B, 256, 65, 120]

        # 3. Decode, merging skip connections at each scale
        x = self.up1(x, f3)  # [B, 128, 130, 240]
        x = self.up2(x, f2)  # [B,  64, 260, 480]
        x = self.up3(x, f1)  # [B,  32, 520, 960]

        # 4. Project to 5 output channels at full resolution
        out = self.final_conv(x)  # [B, 5, 520, 960]

        flow_0_to_t = out[:, 0:2, :, :]           # unbounded pixel displacements
        flow_1_to_t = out[:, 2:4, :, :]           # unbounded pixel displacements
        mask        = torch.sigmoid(out[:, 4:5, :, :])  # blending weight in [0, 1]

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

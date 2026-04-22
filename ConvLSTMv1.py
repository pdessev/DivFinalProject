import torch
import torch.nn as nn

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
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(BiConvLSTM, self).__init__()
        
        self.forward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.backward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

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
            h_f, c_f = self.forward_cell(x[:, t, :, :, :], (h_f, c_f))
            forward_outputs.append(h_f)
            
        # Backward pass
        for t in range(seq_len - 1, -1, -1):
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

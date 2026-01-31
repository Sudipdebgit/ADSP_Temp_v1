import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedBSS(nn.Module):
    """
    Improved CNN-based model for blind source separation.
    Balanced between performance and memory efficiency.
    Uses dilated convolutions and residual connections.
    """
    
    def __init__(self, n_channels=2, n_freq=257, n_sources=2):
        """
        Args:
            n_channels: number of input channels (stereo = 2)
            n_freq: number of frequency bins (n_fft//2 + 1)
            n_sources: number of sources to separate (2 for speech/noise)
        """
        super(ImprovedBSS, self).__init__()
        
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.n_sources = n_sources
        
        # Initial feature extraction - minimal increase from 32 to 34
        self.input_conv = nn.Sequential(
            nn.Conv2d(n_channels, 34, kernel_size=3, padding=1),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale feature extraction with dilated convolutions
        # Very small channel increases: 34->40->42->40->34
        self.dilated_block1 = self._dilated_residual_block(34, 40, dilation=1)
        self.dilated_block2 = self._dilated_residual_block(40, 42, dilation=2)
        self.dilated_block3 = self._dilated_residual_block(42, 40, dilation=4)
        self.dilated_block4 = self._dilated_residual_block(40, 34, dilation=2)
        
        # Feature refinement - keep small
        self.refine = nn.Sequential(
            nn.Conv2d(34, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Mask prediction with residual connection
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_sources * n_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _dilated_residual_block(self, in_channels, out_channels, dilation):
        """
        Dilated convolution block with residual connection.
        Captures multi-scale temporal and frequency patterns.
        """
        return DilatedResidualBlock(in_channels, out_channels, dilation)
    
    def forward(self, mix_mag):
        """
        Args:
            mix_mag: (batch, n_channels, freq, time)
        
        Returns:
            s1_mag: (batch, n_channels, freq, time) - estimated source 1 magnitude
            s2_mag: (batch, n_channels, freq, time) - estimated source 2 magnitude
            masks: (batch, n_sources, n_channels, freq, time) - predicted masks
        """
        batch_size = mix_mag.shape[0]
        
        # Initial features
        x = self.input_conv(mix_mag)
        
        # Multi-scale dilated feature extraction
        x = self.dilated_block1(x)
        x = self.dilated_block2(x)
        x = self.dilated_block3(x)
        x = self.dilated_block4(x)
        
        # Refine features
        x = self.refine(x)
        
        # Predict masks
        masks_flat = self.mask_predictor(x)  # (batch, n_sources * n_channels, freq, time)
        
        # Reshape masks
        masks = masks_flat.view(batch_size, self.n_sources, self.n_channels, self.n_freq, -1)
        
        # Apply masks to mixture
        s1_mask = masks[:, 0, :, :, :]  # (batch, n_channels, freq, time)
        s2_mask = masks[:, 1, :, :, :]
        
        s1_mag = mix_mag * s1_mask
        s2_mag = mix_mag * s2_mask
        
        return s1_mag, s2_mag, masks


class DilatedResidualBlock(nn.Module):
    """
    Residual block with dilated convolution.
    """
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection for residual connection if dimensions change
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.projection is not None:
            identity = self.projection(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class SemiSupervisedBSSLoss(nn.Module):
    """
    Enhanced loss for semi-supervised learning:
    - Supervised: MSE on magnitude spectrograms
    - Unsupervised: Reconstruction consistency
    - Sparsity: Encourage masks to be sparse (optional)
    """
    
    def __init__(self, lambda_unsup=0.1, lambda_sparsity=0.0):
        super(SemiSupervisedBSSLoss, self).__init__()
        self.lambda_unsup = lambda_unsup
        self.lambda_sparsity = lambda_sparsity
    
    def forward(self, pred_s1, pred_s2, target_s1, target_s2, mix_mag, is_supervised):
        """
        Args:
            pred_s1, pred_s2: predicted source magnitudes
            target_s1, target_s2: ground truth source magnitudes
            mix_mag: mixture magnitude
            is_supervised: (batch,) bool tensor indicating supervised samples
        """
        batch_size = pred_s1.shape[0]
        
        # Supervised loss (MSE on labeled data)
        supervised_loss = 0.0
        n_supervised = is_supervised.sum().item()
        
        if n_supervised > 0:
            sup_mask = is_supervised.view(-1, 1, 1, 1)
            
            s1_loss = F.mse_loss(pred_s1 * sup_mask, target_s1 * sup_mask, reduction='sum')
            s2_loss = F.mse_loss(pred_s2 * sup_mask, target_s2 * sup_mask, reduction='sum')
            
            supervised_loss = (s1_loss + s2_loss) / (n_supervised * 2)
        
        # Unsupervised loss (reconstruction consistency on all data)
        reconstruction = pred_s1 + pred_s2
        unsupervised_loss = F.mse_loss(reconstruction, mix_mag)
        
        # Sparsity loss (optional, encourages cleaner separation)
        sparsity_loss = 0.0
        if self.lambda_sparsity > 0:
            sparsity_loss = (pred_s1.abs().mean() + pred_s2.abs().mean()) / 2
        
        # Total loss
        total_loss = supervised_loss + self.lambda_unsup * unsupervised_loss + self.lambda_sparsity * sparsity_loss
        
        return {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'sparsity': sparsity_loss,
            'n_supervised': n_supervised,
        }


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    mix_mag = torch.randn(batch_size, 2, 257, 507).to(device)
    
    s1_mag, s2_mag, masks = model(mix_mag)
    
    print(f"\nInput shape: {mix_mag.shape}")
    print(f"Output s1 shape: {s1_mag.shape}")
    print(f"Output s2 shape: {s2_mag.shape}")
    print(f"Masks shape: {masks.shape}")
    
    # Test loss
    criterion = SemiSupervisedBSSLoss(lambda_unsup=0.1, lambda_sparsity=0.01)
    
    target_s1 = torch.randn_like(s1_mag)
    target_s2 = torch.randn_like(s2_mag)
    is_supervised = torch.tensor([True, True, False, False])
    
    loss_dict = criterion(s1_mag, s2_mag, target_s1, target_s2, mix_mag, is_supervised)
    
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Calculate memory usage
    mem_per_sample = total_params * 4 / (1024**2)  # MB (float32)
    print(f"\nEstimated model memory: {mem_per_sample:.2f} MB")

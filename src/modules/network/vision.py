import torch.nn as nn
import torch
from torchvision.models import squeezenet1_1
from torchvision.models import SqueezeNet1_1_Weights


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert x.size(1) == self.channels, f"Expected {self.channels} channels, but got {x.size(1)}"
        weights = self.fc(x)[:, :, None, None]
        return x * weights


class LocalBranch(nn.Module):
    def __init__(self, in_channels=6, output_dim=256):
        super(LocalBranch, self).__init__()
        base_model = squeezenet1_1(weights=None)
        self.features = base_model.features
        self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.se = SEBlock(channels=in_channels)  # Match channels with input
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.se(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


class GlobalBranch(nn.Module):
    def __init__(self, input_channels=6, output_dim=256, num_heads=8):
        super(GlobalBranch, self).__init__()
        self.vit = None  # Placeholder for the Vision Transformer
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_heads = num_heads

    def initialize(self, input_shape):
        """
        Dynamically initialize the Vision Transformer based on input shape.

        Args:
            input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
        """
        _, channels, height, width = input_shape

        assert channels == self.input_channels, f"Expected {self.input_channels} input channels, got {channels}"

        # Dynamically set image size for the Vision Transformer
        self.img_size = (height, width)
        from timm.models.vision_transformer import VisionTransformer

        # Initialize the Vision Transformer with the exact input dimensions
        self.vit = VisionTransformer(img_size=self.img_size, patch_size=16, embed_dim=self.output_dim,
                                     num_heads=self.num_heads, num_classes=0)
        # Update the patch embedding projection to match the input channels
        old_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(self.input_channels, old_proj.out_channels,
                                              kernel_size=old_proj.kernel_size, stride=old_proj.stride,
                                              padding=old_proj.padding)

    def forward(self, x):
        # Initialize the Vision Transformer dynamically on the first forward pass
        if self.vit is None:
            self.initialize(x.shape)
        return self.vit(x)


class NoiseAwareFusion(nn.Module):
    def __init__(self, dim):
        super(NoiseAwareFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.noise_gate = nn.Linear(dim, 1)

    def forward(self, local_feat, global_feat):
        noise_weight = torch.sigmoid(self.noise_gate(local_feat))
        local_feat = local_feat * noise_weight
        fused_feat, _ = self.attn(local_feat.unsqueeze(1), global_feat.unsqueeze(1), global_feat.unsqueeze(1))
        return fused_feat.squeeze(1)


class LGCAN(nn.Module):
    def __init__(self, in_channels=6, output_dim=256, num_classes=10):
        super(LGCAN, self).__init__()
        self.local_branch = LocalBranch(in_channels, output_dim)
        self.global_branch = GlobalBranch(input_channels=in_channels, output_dim=output_dim)
        self.fusion = NoiseAwareFusion(dim=output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        fused_feat = self.fusion(local_feat, global_feat)
        return self.classifier(fused_feat)

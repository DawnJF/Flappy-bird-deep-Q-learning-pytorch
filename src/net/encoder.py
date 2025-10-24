import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(
            dim=(2, 3)
        )  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class ImageEncoder(nn.Module):
    """Smart pretrained encoder with efficient parameter usage"""

    def __init__(self, bottleneck_dim=256, freeze_backbone=True, num_features=4):
        super().__init__()
        # Use EfficientNet-B0 as backbone (much lighter than ResNet18 but better performance)

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        ).features[:-4]

        # TODO
        output_shape = self.backbone(torch.zeros(1, 3, 128, 128)).shape
        print(f"ImageEncoder Backbone output shape: {output_shape}")

        # Freeze backbone parameters if requested (similar to JAX frozen encoder)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.spatial_embeddings = SpatialLearnedEmbeddings(
            height=output_shape[2],
            width=output_shape[3],
            channel=output_shape[1],
            num_features=num_features,
        )

        # Efficient feature projection with residual connection
        self.feature_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(output_shape[1] * num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # 输入: (B, C, H, W)
        x = self.backbone(x)
        x = self.spatial_embeddings(x)
        x = self.feature_proj(x)
        return x


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.encoder(state)


class EncoderWrapper(nn.Module):
    def __init__(self, image_num, proprio_dim=7):
        super().__init__()
        self.image_num = image_num
        self.proprio_dim = proprio_dim

        self.image_encoder = ImageEncoder(bottleneck_dim=256)
        self.proprio_encoder = ProprioEncoder(input_dim=proprio_dim, output_dim=64)

    def forward(self, observations):
        state = observations["state"]  # 本体感受信息
        image_rgb = observations["rgb"]
        image_wrist = observations["wrist"]
        images = torch.stack([image_rgb, image_wrist], dim=1)  # (B, N, C, H, W)

        B, N, C, H, W = images.shape
        assert N == self.image_num, f"Expected {self.image_num} images, but got {N}"

        image_features = []

        # Extract features from all images
        for i in range(N):
            img = images[:, i, :, :, :]  # Shape: (B, C, H, W)
            img_features = self.image_encoder(img)  # (B, 256)
            image_features.append(img_features)

        image_features = torch.cat(image_features, dim=1)  # Shape: (B, N * image_dim)

        state_features = self.proprio_encoder(state)
        return torch.cat([image_features, state_features], dim=-1)

    def get_out_shape(self, image_shape=128):
        """获取编码器输出的形状"""

        image1 = torch.zeros(1, self.image_num, 3, image_shape, image_shape)
        state = torch.zeros(1, self.proprio_dim)

        observations = {"state": state, "rgb": image1[:, 0], "wrist": image1[:, 1]}
        return self.forward(observations).shape[1]


if __name__ == "__main__":

    encoder = EncoderWrapper(image_num=2, proprio_dim=7)

    print(
        f"parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
    )
    print(f"sum parameters: {sum(p.numel() for p in encoder.parameters())}")

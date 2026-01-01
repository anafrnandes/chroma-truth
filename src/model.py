import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling com MaxPool e DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling e depois DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Usamos Transposed Convolutions para aumentar a imagem
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # x1 é o input que vem de baixo (upsampled)
        # x2 é o input que vem do lado (skip connection)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        # Concatena os mapas de características (Skip Connection!)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColorUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(ColorUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ENCODER
        # Entrada: 1 canal (L)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Base da rede (Bottleneck)

        # DECODER
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Camada final para gerar os 2 canais (ab)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder com Skip Connections (x4, x3, x2, x1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


# Bloco de Teste
if __name__ == "__main__":
    print("A testar a U-Net...")
    # Simular uma imagem de entrada: Batch=1, Canais=1 (L), Altura=256, Largura=256
    dummy_input = torch.randn(1, 1, 256, 256)

    model = ColorUNet(n_channels=1, n_classes=2)
    output = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

    # Verificação: O output tem de ser [1, 2, 256, 256]
    assert output.shape == (1, 2, 256, 256), "Erro nas dimensões da U-Net!"
    print("A arquitetura está correta.")
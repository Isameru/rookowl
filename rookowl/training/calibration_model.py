""" `CalibrationModel` PyTorch class used for finding the chessboard key-point, used to calibrate the camera to uniform space.
    It is a custom (non-standard) network, but also a clear derivative of VGG and ResNet.
    Skip connections are also built differently from ResNet V1/V2.
    There is no adaptive (average pool) sampling in between feature and classifier layers, to support ONNX model export.
    The input image is a grayscale 160x160 (narrowed horizontally due to non-proportional scaling).
    The output is a set of (x,y) uniform coordinates (-1..+1) of 13 chessboard key-points.
    The network is unable to recognize the side of the chessboard, as well as its presence in the input image.
"""

import torch
from rookowl.calibration import CHESSBOARD_KEY_POINTS_PARAM_COUNT

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), padding=(1, 1)):
        super(ResidualBlock, self).__init__()

        self.fc0 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn0 = torch.nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.a0 = torch.nn.ReLU(inplace=True)

        self.fc1 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.a1 = torch.nn.ReLU(inplace=True)

        self.fc2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.a2 = torch.nn.ReLU(inplace=True)

        self.fc3 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn3 = torch.nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.a3 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.a0(x)
        x0 = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.a1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.a2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.a3(x)

        return x + x0


class CalibrationModel(torch.nn.Module):
    def __init__(self):
        super(CalibrationModel, self).__init__()

        features = []

        def add_conv_block(in_channels, out_channels):
            features.append(ResidualBlock(in_channels, out_channels))

        def add_conv_reduce2():
            features.append(torch.nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1))

        add_conv_block(1, 64)
        add_conv_reduce2()
        add_conv_block(64, 64)
        add_conv_reduce2()
        add_conv_block(64, 128)
        add_conv_block(128, 128)
        add_conv_reduce2()
        add_conv_block(128, 256)
        add_conv_block(256, 256)
        add_conv_block(256, 256)
        add_conv_reduce2()
        add_conv_block(256, 512)
        add_conv_block(512, 512)
        add_conv_block(512, 512)
        add_conv_reduce2()

        self.features = torch.nn.Sequential(*features)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(5*5*512, 8*1024, bias=True),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0, inplace=True),

            torch.nn.Linear(8*1024, 8*1024, bias=True),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0, inplace=True),

            torch.nn.Linear(8*1024, 8*1024, bias=True),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0, inplace=True),

            torch.nn.Linear(
                8*1024, CHESSBOARD_KEY_POINTS_PARAM_COUNT, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

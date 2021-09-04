""" Creates a new PyTorch piece recognition model to start training with and saves it to a file.
"""

import argparse
import os

import torch
from rookowl.global_var import PIECE_INPUT_IMAGE_SHAPE
from rookowl.piece import PIECE_CODE_COUNT
from rookowl.utils.common import prod

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument("-m", dest="model_file", default="models/piece_model-new.pth",
                        help="new piece recognition model file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = torch.hub.load('pytorch/vision:v0.9.0',
                           'vgg11_bn', pretrained=True)

    # Replace the first CNN so it accepts a grayscale image (1 input channel).
    model.features[0] = torch.nn.Conv2d(
        1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Infer the output shape of `feature` model part.
    input_shape = (
        1, 1, PIECE_INPUT_IMAGE_SHAPE[0], PIECE_INPUT_IMAGE_SHAPE[1])
    input = torch.randn(size=input_shape, requires_grad=True)
    print(f"Input Shape  : {list(input.shape)}")

    x = input
    for layer in model.features:
        x = layer(x)
        print(f"{x.shape} ({prod(x.shape)}) <- {layer}")
    print()

    features_output = model.features(input)
    print(f"Features Output Shape : {list(features_output.shape)}")

    # Neutralize `AdaptiveAvgPool2d` middle-layer.
    model.avgpool = torch.nn.Identity()

    middle_activation = model.avgpool(features_output)
    # Activations are 512x7x4 at this point.
    print(f"Middle Output Shape : {list(middle_activation.shape)}")

    # Fix classifier NN layers to accept 7x4 input image and produce 13 outputs.
    model.classifier[0] = torch.nn.Linear(in_features=prod(middle_activation.shape),
                                          out_features=4096, bias=True)
    model.classifier[-1] = torch.nn.Linear(in_features=4096,
                                           out_features=PIECE_CODE_COUNT, bias=True)

    output = model(input)
    print(f"Output Shape : {list(output.shape)}")
    print()
    print("Model topology:")
    print(model)
    print()
    print(f"Saving model to file: {args.model_file}")
    torch.save(model, args.model_file)

    model_size = os.path.getsize(args.model_file)
    print(f"Model file size: {model_size}")

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

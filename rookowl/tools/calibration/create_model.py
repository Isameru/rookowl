""" Creates a new PyTorch camera calibration model to start training with and saves it to a file.
"""
import argparse
import os

import torch
from rookowl.global_var import CALIBRATION_INPUT_IMAGE_SHAPE
from rookowl.training.calibration_model import CalibrationModel
from rookowl.utils.common import prod

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", dest="model_file", type=str, default="models/calibration_model-new.pth",
                        help="new calibration model file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = CalibrationModel()

    # Infer the output shape of `feature` model part.
    input_shape = (
        1, 1, CALIBRATION_INPUT_IMAGE_SHAPE[0], CALIBRATION_INPUT_IMAGE_SHAPE[1])
    input = torch.randn(size=input_shape, requires_grad=True)
    print(f"Input Shape  : {list(input.shape)}")

    x = input
    for layer in model.features:
        x = layer(x)
        print(f"{x.shape} ({prod(x.shape)}) <- {layer}")
    print()

    features_output = model.features(input)
    print(f"Features Output Shape : {list(features_output.shape)}")

    output = model(input)
    print(f"Output Shape : {list(output.shape)}")
    print()
    print("Model topology:")
    print(model)
    print()
    print(f"Saving model to file: {args.model_file}")
    torch.save(model.state_dict(), args.model_file)

    model_size = os.path.getsize(args.model_file)
    print(f"Model file size: {model_size}")

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

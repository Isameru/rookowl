""" Converts the calibration PyTorch model to TensorFlow Lite.
"""

import argparse
import os

import torch
from rookowl.global_var import CALIBRATION_INPUT_IMAGE_SHAPE
from rookowl.training.calibration_model import CalibrationModel
from rookowl.utils.model_conversion import convert_pt_model_to_tflite

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", dest="model_file", default="models/calibration_model-best.pth",
                        help="model file path to convert")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = CalibrationModel()
    model.load_state_dict(torch.load(args.model_file))
    basename = os.path.splitext(args.model_file)[0]
    input_shape = (
        1, 1, CALIBRATION_INPUT_IMAGE_SHAPE[0], CALIBRATION_INPUT_IMAGE_SHAPE[1])

    tflite_model_file = convert_pt_model_to_tflite(
        model, basename, input_shape, f"{basename}.tflite")

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

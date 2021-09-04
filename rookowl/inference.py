""" ...
"""

# !!! Under construction

import os

import cv2 as cv
import numpy as np

from rookowl.global_var import (CALIBRATION_INPUT_IMAGE_SHAPE,
                                PIECE_INPUT_IMAGE_SHAPE)
from rookowl.piece import PIECE_TEXT_MAP
from rookowl.piece_image import segment_pieces_from_image

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def make_model_interpreter(model_file: str):
    extension = os.path.splitext(model_file)[-1]
    if extension == "":
        return PyTorchModelInterpreter(model_file)
    elif extension == ".tflite":
        return TensorFlowLiteModelInterpreter(model_file)
    else:
        raise Exception(
            f"Unrecognized ML framework model based on file extension: {extension}")

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class CalibrationRecognizer:
    def __init__(self, model_file: str):
        assert os.path.isfile(model_file)
        self.interpreter = make_model_interpreter(model_file)

    def infer_crosspoints(self, image):
        if not all(image.shape == CALIBRATION_INPUT_IMAGE_SHAPE):
            image = cv.resize(
                image, (CALIBRATION_INPUT_IMAGE_SHAPE[1], CALIBRATION_INPUT_IMAGE_SHAPE[0]), interpolation=cv.INTER_CUBIC)

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            image = (image - 0.456) / 0.224

        # Add a single grayscale channel.
        image = np.expand_dims(image, axis=0)
        # Add a mini-batch of 1 sample.
        image = np.expand_dims(image, axis=0)

        return np.reshape(self.interpreter.infer(image), (-1, 2))

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class PieceRecognizer:
    def __init__(self, model_file: str):
        assert os.path.isfile(model_file)
        self.interpreter = make_model_interpreter(model_file)
        self.piece_images = None

    def set_source(self, image, crosspoints):
        self.piece_images = segment_pieces_from_image(image, crosspoints)
        for piece_index, piece_image in enumerate(self.piece_images):
            assert all(piece_image.shape == PIECE_INPUT_IMAGE_SHAPE)
            # Add a single grayscale channel.
            piece_image = np.expand_dims(piece_image, axis=0)
            # Add a mini-batch of 1 sample.
            self.piece_images[piece_index] = np.expand_dims(
                piece_image, axis=0)

    def infer_piece(self, square_index):
        assert self.piece_images is not None

        image = self.piece_images[square_index]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            image = (image - 0.456) / 0.224

        return self.interpreter.infer(image)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def recognize_framework_from_model_file(model_file: str):
    # Remove
    assert os.path.isfile(model_file)
    with open(model_file, "rb") as file:
        header = file.read(2)
        if header == b"PK":
            return "pytorch"
        elif header == b"$\x00":
            return "tensorflow-lite"
        else:
            raise Exception("Unrecognized ML framework model")


class ChessboardRecognizer:
    # Update
    def __init__(self, calibration_model_file: str, piece_model_file: str):
        self.framework = recognize_framework_from_model_file(
            calibration_model_file)
        print(f"Recognized ML framework: {self.framework}")
        assert recognize_framework_from_model_file(
            piece_model_file) == self.framework, "Both calibration and piece recognition models must run with the same ML framework."

        if self.framework == "pytorch":
            ModelInterpreter = PyTorchModelInterpreter
        elif self.framework == "tensorflow-lite":
            ModelInterpreter = TensorFlowLiteModelInterpreter

        print("Initializing chessboard calibration model interpreter...")
        self.calibration_interpreter = ModelInterpreter(calibration_model_file)
        print("Initializing piece recognition model interpreter...")
        self.piece_interpreter = ModelInterpreter(piece_model_file)

    def infer_crosspoints(self, image):
        image = cv.resize(
            image, (CALIBRATION_INPUT_IMAGE_SHAPE[1], CALIBRATION_INPUT_IMAGE_SHAPE[0]), interpolation=cv.INTER_CUBIC)
        # mean, std_dev = cv.meanStdDev(image)
        # print(mean, std_dev)

        image = image.astype(np.float32) / 255.0
        image = (image - 0.456) / 0.224

        # Add a single grayscale channel.
        image = np.expand_dims(image, axis=0)
        # Add a mini-batch of 1 sample.
        image = np.expand_dims(image, axis=0)

        return np.reshape(self.calibration_interpreter.infer(image), (-1, 2))

    def infer_pieces(self, image, crosspoints):
        piece_images = segment_pieces_from_image(
            image, crosspoints, augment=False)
        for piece_index, piece_image in enumerate(piece_images):
            assert all(piece_image.shape == PIECE_INPUT_IMAGE_SHAPE)
            piece_images[piece_index] = np.expand_dims(piece_image, axis=0)
        piece_images = np.stack(piece_images, axis=0)

        piece_images = piece_images.astype(np.float32) / 255.0
        piece_images = (piece_images - 0.456) / 0.224

        return self.piece_interpreter.infer(piece_images)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class PyTorchModelInterpreter:
    def __init__(self, model_file: str):
        import torch
        self.torch = torch
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.model = self.torch.load(model_file).to(self.device)
        self.model.eval()

    def infer(self, input):
        input = self.torch.from_numpy(input).to(self.device)
        with self.torch.no_grad():
            return self.model(input).cpu().numpy()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class TensorFlowLiteModelInterpreter:
    def __init__(self, model_file: str):
        import tflite_runtime.interpreter as tflite
        self.tflite = tflite
        self.interpreter = tflite.Interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, input):
        self.interpreter.set_tensor(self.input_details[0]["index"], input)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


if __name__ == "__main__":
    #image = cv.imread("dataset/eval/test.jpg", cv.IMREAD_GRAYSCALE)
    image = cv.imread("/home/pi/test.jpg", cv.IMREAD_GRAYSCALE)

    calib_reco = CalibrationRecognizer(
        "/home/pi/rookowl/model_tflite/calibration_model_best.tflite")
    import time
    s = time.clock()
    y = calib_reco.infer_crosspoints(image)
    e = time.clock()
    print(e-s)
    print(y)

    # reco = ChessboardRecognizer(
    #     "rookowl/model_tflite/calibration_model_best.tflite", "rookowl/model_tflite/piece_model_best.tflite")

    # y = reco.infer_crosspoints(image)
    # z = reco.infer_pieces(image, y)

    # piece_texts = list(PIECE_TEXT_MAP.values())
    # z = np.argmax(z, axis=1)
    # for i, z in enumerate(z):
    #     print(piece_texts[z])
    #     if (i % 8) == 7:
    #         print()

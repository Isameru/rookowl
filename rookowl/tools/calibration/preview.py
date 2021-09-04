""" Tool for previewing the calibration of images prepared for the training or evaluating the
    calibration from the unaltered photos.
"""

import argparse
import random

import cv2 as cv
import numpy as np
import torch
from rookowl.calibration import (CHESSBOARD_KEY_POINTS, calibrate_from_points,
                                 draw_chessboard_cross_points)
from rookowl.training.calibration_dataset import CalibrationDataset
from rookowl.training.calibration_model import CalibrationModel

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class App:
    MAIN_WIN_NAME = "Calibration Preview Tool"

    def __init__(self, label_dir: str, stockphotos_dir: str, model_file: str, purpose: str):
        assert purpose in ["train", "eval"]
        self.dataset = CalibrationDataset(
            label_dir, augment=(purpose == "train"), stockphotos_dir=stockphotos_dir, preview=True)

        try:
            print(f"Loading model: {model_file}")
            self.model = CalibrationModel()
            self.model.load_state_dict(torch.load(model_file))
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.model.eval()
        except Exception as ex:
            print(f"Failed to load the model: {str(ex)}")
            self.model = None

    def run(self):
        cv.namedWindow(self.MAIN_WIN_NAME)

        while True:
            index = random.randint(0, len(self.dataset)-1)
            sample = self.dataset[index]

            if self.model is not None:
                input_batch = sample["image"].unsqueeze(0)

                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')

                with torch.no_grad():
                    output = self.model(input_batch)[0]

                output = np.reshape(output.cpu().numpy(), (-1, 2))

                calibration, rms_error = calibrate_from_points(
                    CHESSBOARD_KEY_POINTS, output)
            else:
                calibration = None

            image = sample["image_preview"].copy()

            # Draw the human-labelled reference cross-points.
            draw_chessboard_cross_points(image, sample["calibration"])

            # Draw the inferred cross-points as white dots.
            if calibration is not None:
                draw_chessboard_cross_points(
                    image, calibration, color=(0, 255, 0))

            # Display the image.
            cv.imshow(self.MAIN_WIN_NAME, image)

            key = cv.waitKey(0) & 0xff
            if key == 27:
                break

        cv.destroyAllWindows()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", type=str, help="train, eval")
    parser.add_argument("-d", dest="dataset_dir", type=str, default="dataset/validation",
                        help="dataset directory path, where chessboard images and their labels are placed")
    parser.add_argument("-s", dest="stockphotos_dir", type=str,
                        default="stockphotos", help="stock photos directory path")
    parser.add_argument("-m", dest="model_file", type=str,
                        default="models/calibration_model-best.pth", help="model file path to load")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = App(args.dataset_dir, args.stockphotos_dir,
              args.model_file, args.purpose)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

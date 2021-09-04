""" Contains `CalibrationDataset` class - a PyTorch dataset - and related utilities to cut the
    chessboard silhouette out of the photograph and compose training samples.
"""

import random

import cv2 as cv
import numpy as np
import torch
from rookowl.calibration import (CHESSBOARD_KEY_POINTS, calibrate_from_points,
                                 project_points)
from rookowl.calibration_image import augment_calibration_train_image
from rookowl.global_var import (CALIBRATION_DISPLAY_SHAPE,
                                CALIBRATION_INPUT_IMAGE_SHAPE,
                                CALIBRATION_PREPROCESSING_NORMALIZATION)
from rookowl.label import load_labels, search_stockphotos
from torch.utils.data import Dataset
from torchvision import transforms

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def generate_train_example(label, background_filepath, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    # Load the labelled image containing the chessboard.
    photo = cv.imread(label["image_filepath"], cv.IMREAD_GRAYSCALE)

    # Point out all the chessboard cross-section points within the image.
    board_coords = project_points(
        label["calibration"], CHESSBOARD_KEY_POINTS)

    # Load the background image.
    background_image = cv.imread(background_filepath, cv.IMREAD_GRAYSCALE)

    # Generate a randomized training example image with all the points transformed accordingly.
    image, image_points = augment_calibration_train_image(
        photo, board_coords, background_image)

    # Build the calibration information for the newly generated image.
    calibration, _ = calibrate_from_points(
        CHESSBOARD_KEY_POINTS, image_points)

    return image, calibration, image_points

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class CalibrationDataset(Dataset):
    def __init__(self, label_dir: str, augment: bool = False, stockphotos_dir: str = None, preview: bool = False):
        self.labels = load_labels(
            label_dir, with_state_only=False, with_pieces_only=False)
        self.length = len(self.labels)
        self.augment = augment
        if augment:
            assert stockphotos_dir is not None, "Augmentation requires stockphoto library"
            self.stockphoto_paths = search_stockphotos(stockphotos_dir)
        self.preview = preview

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.augment:
            image, calibration, image_points = generate_train_example(
                label, random.choice(self.stockphoto_paths))
            if self.preview:
                image_preview = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                image_preview = cv.resize(
                    image_preview, (CALIBRATION_DISPLAY_SHAPE[1], CALIBRATION_DISPLAY_SHAPE[0]), interpolation=cv.INTER_NEAREST)
        else:
            image = cv.imread(label["image_filepath"], cv.IMREAD_GRAYSCALE)
            image = cv.resize(
                image, (CALIBRATION_INPUT_IMAGE_SHAPE[1], CALIBRATION_INPUT_IMAGE_SHAPE[0]), interpolation=cv.INTER_LINEAR)
            calibration = label["calibration"]
            image_points = project_points(calibration, CHESSBOARD_KEY_POINTS)

            if self.preview:
                image_preview = cv.imread(label["image_filepath"])
                image_preview = cv.resize(
                    image_preview, (CALIBRATION_DISPLAY_SHAPE[1], CALIBRATION_DISPLAY_SHAPE[0]), interpolation=cv.INTER_LINEAR)

        image = self.preprocess(image)

        sample = {
            "image": image,
            "image_points": np.reshape(image_points, (-1,)),
            "calibration": calibration
        }

        if self.preview:
            sample["image_preview"] = image_preview

        return sample

    def preprocess(self, image):
        assert image.dtype == np.uint8
        assert all(image.shape == CALIBRATION_INPUT_IMAGE_SHAPE)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.type(torch.float32) / 255.0
        image = transforms.Normalize(
            **CALIBRATION_PREPROCESSING_NORMALIZATION)(image)
        return image

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

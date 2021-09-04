""" Contains `PieceDataset` class and related utilities to cut the chessboard photograph into 64
    images for each chessboard piece or square.
"""

import random

import cv2 as cv
import numpy as np

from rookowl.calibration import CHESSBOARD_CROSS_POINTS, project_points
from rookowl.global_var import PIECE_INPUT_IMAGE_SHAPE
from rookowl.utils.image import coord_vec_to_px, crop_image

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

PIECE_HORIZONTAL_MARGIN_RATIO = 1.25

PIECE_AUGMENTATION_INPUT_NOISE_INT8 = 4
PIECE_AUGMENTATION_OUTPUT_NOISE_INT8 = 8
PIECE_AUGMENTATION_HORIZONTAL_MARGIN_RATIO_RANGE = (1.0, 1.6)
PIECE_AUGMENTATION_BOTTOM_MARGIN_RATIO_RANGE = (0.9, 1.13)
PIECE_AUGMENTATION_ROTATION_RANGE = (-10.5, +10.5)
PIECE_AUGMENTATION_SCALE_ABS_ALPHA_RANGE = (0.8, 1.25)
PIECE_AUGMENTATION_SCALE_ABS_BETA_RANGE = (-45, +45)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def segment_pieces_from_image(image, image_cross_points, augment: bool = False):
    """ Given an image and all 9x9=91 image chessboard crosspoints, loads and segments the
        chessborad photo extracting 64 images of all squares (empty or with pieces).
    """
    assert image.dtype == np.uint8
    assert len(image.shape) == 2, "Image must be in grayscale"
    assert len(image_cross_points) == len(CHESSBOARD_CROSS_POINTS)

    if augment:
        # Noise the input image.
        noise = np.zeros(image.shape, np.int8)
        cv.randn(noise, 0, random.randint(
            0, PIECE_AUGMENTATION_INPUT_NOISE_INT8))
        image = cv.add(image, noise, dtype=cv.CV_8UC3)
        del noise

    # Transform the image points from -1...1 coordinates to pixels.
    image_cross_points_px = coord_vec_to_px(image.shape, image_cross_points)

    piece_images = []
    piece_input_image_shape_ratio = PIECE_INPUT_IMAGE_SHAPE[0] / \
        PIECE_INPUT_IMAGE_SHAPE[1]

    for square_y in range(8):
        for square_x in range(8):
            square_indices = [
                (9 * square_y) + square_x,
                (9 * square_y) + square_x + 1,
                (9 * (square_y + 1)) + square_x + 1,
                (9 * (square_y + 1)) + square_x,
            ]

            crop_x, crop_y, crop_w, crop_h = cv.boundingRect(
                np.array([image_cross_points_px[i] for i in square_indices], dtype=np.int32))
            crop_center = np.array(
                [crop_x + crop_w / 2, crop_y + crop_h / 2], dtype=np.float32)

            dist_left_px = crop_center[0] - crop_x
            dist_right_px = crop_x + crop_w - crop_center[0]
            dist_down_px = crop_y + crop_h - crop_center[1]

            dist_down_px = crop_w // 2

            if augment:
                dist_left_px *= random.uniform(
                    *PIECE_AUGMENTATION_HORIZONTAL_MARGIN_RATIO_RANGE)
                dist_right_px *= random.uniform(
                    *PIECE_AUGMENTATION_HORIZONTAL_MARGIN_RATIO_RANGE)
                dist_down_px *= random.uniform(
                    *PIECE_AUGMENTATION_BOTTOM_MARGIN_RATIO_RANGE)
            else:
                dist_left_px *= PIECE_HORIZONTAL_MARGIN_RATIO
                dist_right_px *= PIECE_HORIZONTAL_MARGIN_RATIO

            crop_x = int(crop_center[0] - dist_left_px)
            crop_w = int(dist_right_px + dist_left_px)
            crop_h = int(crop_w * piece_input_image_shape_ratio)
            crop_y = int(crop_center[1] + dist_down_px - crop_h)

            border_color = random.randint(0, 255) if augment else 128
            piece_image, _ = crop_image(
                image, [], (crop_x, crop_y, crop_w, crop_h), border_color)

            if augment:
                # Randomly flip the background (horizontally).
                if random.choice([False, True]):
                    piece_image = cv.flip(
                        piece_image, 1)

                # Rotate the piece image by a random degree left or right.
                degree = random.uniform(*PIECE_AUGMENTATION_ROTATION_RANGE)
                pivot = (crop_w // 2, crop_h - crop_w // 2)
                rotation_matrix = cv.getRotationMatrix2D(pivot, degree, 1.0)
                piece_image = cv.warpAffine(piece_image, rotation_matrix,
                                            (piece_image.shape[1], piece_image.shape[0]), borderValue=border_color)

            piece_image = cv.resize(piece_image, (PIECE_INPUT_IMAGE_SHAPE[1], PIECE_INPUT_IMAGE_SHAPE[0]),
                                    interpolation=cv.INTER_CUBIC)

            if augment:
                # Randomize the piece image's brightness and contrast.
                piece_image = cv.convertScaleAbs(
                    piece_image,
                    alpha=random.uniform(
                        *PIECE_AUGMENTATION_SCALE_ABS_ALPHA_RANGE),
                    beta=random.uniform(*PIECE_AUGMENTATION_SCALE_ABS_BETA_RANGE))

                # Noise the final piece image.
                noise = np.zeros(piece_image.shape, np.int8)
                cv.randn(noise, 0, random.randint(
                    0, PIECE_AUGMENTATION_OUTPUT_NOISE_INT8))
                piece_image = cv.add(piece_image, noise, dtype=cv.CV_8UC3)

            piece_images.append(piece_image)

    return piece_images


def segment_pieces_from_label(label: dict, augment: bool = False):
    """ Given a label, loads and segments the corresponding chessborad photo,
        and extracts 64 images of all squares (empty or with pieces).
    """
    image = cv.imread(label["image_filepath"], cv.IMREAD_GRAYSCALE)
    image_points = project_points(
        label["calibration"], CHESSBOARD_CROSS_POINTS)
    return segment_pieces_from_image(image, image_points, augment)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

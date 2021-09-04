""" Contains `CalibrationDataset` class - a PyTorch dataset - and related utilities to cut the
    chessboard silhouette out of the photograph and compose training samples.
"""

import random

import cv2 as cv
import numpy as np

from rookowl.calibration import CHESSBOARD_KEY_POINTS
from rookowl.global_var import CALIBRATION_INPUT_IMAGE_SHAPE
from rookowl.utils.image import (coord_vec_from_px, coord_vec_to_px,
                                 crop_image, draw_random_blobs,
                                 generate_contour_mask, noise_image,
                                 paste_image, rotate_image, scale_image)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

CALIBRATION_PERFECT_FINETUNING_FILL_RATIO = 0.85

CALIBRATION_AUGMENTATION_INPUT_NOISE_INT8 = 2
CALIBRATION_AUGMENTATION_OUTPUT_NOISE_INT8 = 4
CALIBRATION_AUGMENTATION_FINETUNING_FILL_RATIO_RANGE = (
    0.925*CALIBRATION_PERFECT_FINETUNING_FILL_RATIO, CALIBRATION_PERFECT_FINETUNING_FILL_RATIO/0.925)
CALIBRATION_AUGMENTATION_FILL_RATIO_RANGE = (0.6, 1.2)
CALIBRATION_AUGMENTATION_ROTATION_RANGE = (-9, +9)
CALIBRATION_AUGMENTATION_OFFSCREEN_FACTOR = 0.3
CALIBRATION_AUGMENTATION_FINETUNING_SILHOUETTE_MARGIN_RATIO_RANGE = (1.0, 1.15)
CALIBRATION_AUGMENTATION_SILHOUETTE_MARGIN_RATIO_RANGE = (1.0, 1.5)
CALIBRATION_AUGMENTATION_OUTER_MASK_TRANSPARENCY_RANGE = (0.2, 1.0)
CALIBRATION_AUGMENTATION_CHESSBOARD_SCALE_ABS_ALPHA_RANGE = (0.8, 1.25)
CALIBRATION_AUGMENTATION_CHESSBOARD_SCALE_ABS_BETA_RANGE = (-40, +40)
CALIBRATION_AUGMENTATION_BACKGROUND_SCALE_ABS_ALPHA_RANGE = (0.8, 1.25)
CALIBRATION_AUGMENTATION_BACKGROUND_SCALE_ABS_BETA_RANGE = (-40, +40)
CALIBRATION_AUGMENTATION_MAX_BLOB_COUNT = 1
CALIBRATION_AUGMENTATION_BLOB_PROBABILITY = 0.5
CALIBRATION_AUGMENTATION_BLOB_SIZE_FACTOR = 6

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def augment_calibration_train_image(image, image_points, background_image, finetuning=None):
    """ Augments the specified `image` and composes a training image for the calibration model, by merging it with the `background_image`.
        The image must be accompanied with pixel coordinates `image_points_px` of chessboard key-points (there are exactly 13).
        The resulting image must be further pre-processed (normalized) before inserting into the model.
    """
    assert len(image_points) == len(CHESSBOARD_KEY_POINTS)

    if finetuning is None:
        finetuning = random.choice([False, True])

    # Noise the input image.
    image = noise_image(image, random.randint(
        0, CALIBRATION_AUGMENTATION_INPUT_NOISE_INT8))

    # Noise the input background image.
    background_image = noise_image(background_image, random.randint(
        0, CALIBRATION_AUGMENTATION_INPUT_NOISE_INT8))

    # Shrink the input image to speed up the augmentation process.
    image = cv.resize(
        image, (CALIBRATION_INPUT_IMAGE_SHAPE[1], CALIBRATION_INPUT_IMAGE_SHAPE[0]), interpolation=cv.INTER_LINEAR)

    # Shrink the input background image.
    background_image = cv.resize(
        background_image, (CALIBRATION_INPUT_IMAGE_SHAPE[1], CALIBRATION_INPUT_IMAGE_SHAPE[0]), interpolation=cv.INTER_LINEAR)

    # Randomly flip the background (horizontally, vertically or both).
    bg_flip_throw = random.choice([None, -1, 0, 1])
    if bg_flip_throw is not None:
        background_image = cv.flip(background_image, bg_flip_throw)

    # Transform the image points from -1...1 coordinates to pixels.
    image_points_px = coord_vec_to_px(image.shape, image_points)

    def center_px():
        return image_points_px[len(image_points_px) // 2]

    # Randomly flip the input image (horizontally).
    if random.choice([False, True]):
        image = cv.flip(image, 1)
        image_points_px = [(image.shape[1] - p[0] - 1, p[1])
                           for p in image_points_px]
        image_points_px = [image_points_px[i]
                           for i in [2, 1, 0, 4, 3, 7, 6, 5, 9, 8, 12, 11, 10]]

    # Rotate the cropped image by a random degree left or right.
    image, image_points_px = rotate_image(
        image, image_points_px, center_px(), random.uniform(*CALIBRATION_AUGMENTATION_ROTATION_RANGE))

    # Try to enlarge the silhouette of the chessboard, making sure it all fits in the source image.
    corners_offsets = [(image_points_px[corner_index] - center_px())
                       for corner_index in [0, 2, 10, 12]]
    corners_px = [(center_px() + c) for c in corners_offsets]

    if finetuning:
        try_size_ratio = random.uniform(
            *CALIBRATION_AUGMENTATION_FINETUNING_SILHOUETTE_MARGIN_RATIO_RANGE)
    else:
        try_size_ratio = random.uniform(
            *CALIBRATION_AUGMENTATION_SILHOUETTE_MARGIN_RATIO_RANGE)

    for try_count in range(10):
        enlarged_corner_px = [(center_px() + try_size_ratio * o)
                              for o in corners_offsets]
        fits = all([(p[0] >= 0 and p[1] >= 0 and p[0] <= (image.shape[1]-1)
                   and (p[1] <= image.shape[0]-1)) for p in enlarged_corner_px])
        if fits:
            corners_px = enlarged_corner_px
            break
        else:
            try_size_ratio *= 0.95

    # Add 4 additional auxiliary corners to the image point list.
    image_points_px.insert(0, corners_px[0])
    image_points_px.insert(1, corners_px[1])
    image_points_px.append(corners_px[2])
    image_points_px.append(corners_px[3])
    del corners_px

    def corners_px():
        return [image_points_px[1], image_points_px[0], image_points_px[-2], image_points_px[-1]]

    # Crop the chessboard silhouette.
    image, image_points_px = crop_image(
        image, image_points_px, cv.boundingRect(np.array(corners_px(), dtype=np.int32)))

    # Scale the cropped image so it fits in the desired input shape.
    prop_x_to_y_dst = CALIBRATION_INPUT_IMAGE_SHAPE[1] / \
        CALIBRATION_INPUT_IMAGE_SHAPE[0]
    prop_x_to_y_src = image.shape[1] / image.shape[0]

    if finetuning:
        fill_ratio = random.uniform(
            *CALIBRATION_AUGMENTATION_FINETUNING_FILL_RATIO_RANGE)
    else:
        fill_ratio = random.uniform(*CALIBRATION_AUGMENTATION_FILL_RATIO_RANGE)

    if prop_x_to_y_dst >= prop_x_to_y_src:
        dst_h = CALIBRATION_INPUT_IMAGE_SHAPE[0] * fill_ratio
        dst_w = dst_h * prop_x_to_y_src
    else:
        dst_w = CALIBRATION_INPUT_IMAGE_SHAPE[1] * fill_ratio
        dst_h = dst_w / prop_x_to_y_src
    dst_w = round(dst_w)
    dst_h = round(dst_h)

    image, image_points_px = scale_image(
        image, image_points_px, (round(dst_h), round(dst_w)))

    # Create a new image and paste the cropped chessboard image within its borders in random position.
    new_image = np.zeros(
        (CALIBRATION_INPUT_IMAGE_SHAPE[0], CALIBRATION_INPUT_IMAGE_SHAPE[1]), dtype=image.dtype)

    def center_px():
        return image_points_px[(len(image_points_px)-1) // 2]

    if finetuning:
        offset = (int(new_image.shape[1] - dst_w) // 2,
                  int(new_image.shape[0] - dst_h) // 2)
    else:
        offset = (
            random.randint(0 - int(CALIBRATION_AUGMENTATION_OFFSCREEN_FACTOR * dst_w/2),
                           new_image.shape[1] - dst_w + int(CALIBRATION_AUGMENTATION_OFFSCREEN_FACTOR * (dst_w - dst_w/2))),
            random.randint(0 - int(CALIBRATION_AUGMENTATION_OFFSCREEN_FACTOR * center_px()[1]),
                           new_image.shape[0] - dst_h + int(CALIBRATION_AUGMENTATION_OFFSCREEN_FACTOR * (dst_h - dst_h/2))))

    image_points_px = paste_image(new_image, image, image_points_px, offset)

    # Randomize the new image's brightness and contrast.
    new_image = cv.convertScaleAbs(new_image, alpha=random.uniform(
        *CALIBRATION_AUGMENTATION_CHESSBOARD_SCALE_ABS_ALPHA_RANGE), beta=random.uniform(*CALIBRATION_AUGMENTATION_CHESSBOARD_SCALE_ABS_BETA_RANGE))

    # Generate the silhouette mask for the new image.
    mask_outer = generate_contour_mask(new_image.shape, corners_px())
    mask_inner = generate_contour_mask(
        new_image.shape, [image_points_px[0+2], image_points_px[2+2], image_points_px[12+2], image_points_px[10+2]])

    outer_mask_transparency = random.uniform(
        *CALIBRATION_AUGMENTATION_OUTER_MASK_TRANSPARENCY_RANGE)
    mask = (outer_mask_transparency * mask_outer) + \
        ((1.0 - outer_mask_transparency) * mask_inner)
    del mask_inner
    del mask_outer

    # Remove auxiliary corners from the image point list.
    image_points_px = image_points_px[2:len(image_points_px)-2]

    # Randomize the background image brightness and contrast.
    background_image = cv.convertScaleAbs(background_image, alpha=random.uniform(
        *CALIBRATION_AUGMENTATION_BACKGROUND_SCALE_ABS_ALPHA_RANGE), beta=random.uniform(*CALIBRATION_AUGMENTATION_BACKGROUND_SCALE_ABS_BETA_RANGE))

    # Merge the chessboard image with the background using the mask.
    final_image = (new_image * mask) + (background_image * (1 - mask))
    final_image = final_image.astype(np.uint8)

    # Draw random blobs.
    draw_random_blobs(final_image,
                      max_blob_count=CALIBRATION_AUGMENTATION_MAX_BLOB_COUNT,
                      blob_probability=CALIBRATION_AUGMENTATION_BLOB_PROBABILITY,
                      blob_size_divider=CALIBRATION_AUGMENTATION_BLOB_SIZE_FACTOR)

    # Noise the final image.
    final_image = noise_image(final_image, random.randint(
        0, CALIBRATION_AUGMENTATION_OUTPUT_NOISE_INT8))

    # Transform the new image's key point coordinates from pixels back to -1...1 range.
    image_points = coord_vec_from_px(final_image.shape, image_points_px)

    return final_image, image_points

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def tune_image_from_points(image, image_points):
    image_points_px = coord_vec_to_px(image.shape, image_points)

    fill_ratio = CALIBRATION_PERFECT_FINETUNING_FILL_RATIO
    rect_x, rect_y, rect_w, rect_h = cv.boundingRect(
        np.array(image_points_px, dtype=np.int32))

    rect_x = int(rect_x - rect_w * (1 - fill_ratio) / 2)
    rect_y = int(rect_y - rect_h * (1 - fill_ratio) / 2)
    rect_w = int(rect_w * (2 - fill_ratio))
    rect_h = int(rect_h * (2 - fill_ratio))

    prop_x_to_y_dst = image.shape[1] / image.shape[0]
    prop_x_to_y_src = rect_w / rect_h

    if prop_x_to_y_dst >= prop_x_to_y_src:
        dst_h = rect_h
        dst_w = round(dst_h * prop_x_to_y_dst)
    else:
        dst_w = rect_w
        dst_h = round(dst_w / prop_x_to_y_dst)

    if dst_w != rect_w:
        delta = dst_w - rect_w
        rect_x = int(rect_x - delta / 2)
        rect_w = int(rect_w + delta)
    if dst_h != rect_h:
        delta = dst_h - rect_h
        rect_y = int(rect_y - delta / 2)
        rect_h = int(rect_h + delta)

    image, image_points_px = crop_image(
        image, image_points_px, (rect_x, rect_y, rect_w, rect_h), border_color=128)

    image, image_points_px = scale_image(
        image, image_points_px, CALIBRATION_INPUT_IMAGE_SHAPE)

    image_points = coord_vec_from_px(image.shape, image_points_px)

    # TODO: Return any piece of info so the transformation may be reversed later.
    return image, image_points

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

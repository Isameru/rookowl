""" Provides `Calibration` abstraction with means of linking 2D image points with 3D chessboard points.
"""

from collections import namedtuple

import cv2 as cv
import numpy as np

from rookowl.utils.image import coord_to_px

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

# List of chessboard 3D object points used as a key guideline for a labelling tool.
CHESSBOARD_LABEL_POINTS = np.array([
    # Center
    (0.0, 0.0, 0.0),
    # Quarter Mid-Points
    (-0.5, 0.0, +0.5),
    (+0.5, 0.0, +0.5),
    (+0.5, 0.0, -0.5),
    (-0.5, 0.0, -0.5),
    # Corners
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
], dtype=np.float32)

# List of all 9x9 chessboard 3D object points placed on the chessboard cross-points.
CHESSBOARD_CROSS_POINTS = np.array(
    [(-1.0 + x / 4.0, 0.0, -1.0 + y / 4.0,) for y in range(9) for x in range(9)], dtype=np.float32)

# List of indices of those cross-points, which are a key guideline for training and inference from a photo.
CHESSBOARD_KEY_POINT_INDICES = [
    0, 4, 8, 20, 24, 36, 40, 44, 56, 60, 72, 76, 80]
# List of chessboard 3D object points, which are a key guideline for training and inference from a photo.
CHESSBOARD_KEY_POINTS = np.array(
    [CHESSBOARD_CROSS_POINTS[i] for i in CHESSBOARD_KEY_POINT_INDICES], dtype=np.float32)

CHESSBOARD_KEY_POINTS_PARAM_COUNT = len(CHESSBOARD_KEY_POINTS) * 2

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

Calibration = namedtuple(
    "Calibration", ["rvec", "tvec", "camera_matrix", "dist_coeffs"])

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def calibrate_from_points(object_key_points, image_key_points):
    camera_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    dist_coeffs = None

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs, _ = cv.calibrateCameraRO(
        [object_key_points], [image_key_points], (1000, 1000), 0, camera_matrix, dist_coeffs, flags=cv.CALIB_USE_INTRINSIC_GUESS)

    return Calibration(rvecs[0].T[0], tvecs[0].T[0], camera_matrix, dist_coeffs[0]), rms_error


def project_points(calibration: Calibration, object_points):
    (image_points, jacobian) = cv.projectPoints(
        object_points, calibration.rvec, calibration.tvec, calibration.camera_matrix, calibration.dist_coeffs)
    image_points = np.array([p[0] for p in image_points], dtype=np.float32)
    return image_points

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def draw_chessboard_grid(image, calibration, color=(0, 0, 255), thickness=1):
    image_points = project_points(calibration, CHESSBOARD_CROSS_POINTS)

    for y in range(9):
        cv.line(image, coord_to_px(image.shape, image_points[y * 9]),
                coord_to_px(image.shape, image_points[y * 9 + 8]), color, thickness)
    for x in range(9):
        cv.line(image, coord_to_px(image.shape, image_points[x]),
                coord_to_px(image.shape, image_points[x + 8 * 9]), color, thickness)


def draw_chessboard_cross_points(image, calibration, color=(0, 0, 255), thickness=2, radius=2, object_points=CHESSBOARD_KEY_POINTS):
    image_points = project_points(calibration, object_points)

    for image_point in image_points:
        image_point_px = coord_to_px(image.shape, image_point)
        if any(image_point_px < np.array([-radius, -radius])) or any(image_point_px > np.array([image.shape[1]+radius, image.shape[1]+radius])):
            continue
        cv.circle(image, image_point_px, radius, color, thickness)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

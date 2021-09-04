""" Project-wide global constants.
"""

import numpy as np

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
DOWNLOAD_STOCKPHOTO_SHAPE = np.array((600, 800), dtype=np.int32)
CALIBRATION_DISPLAY_SHAPE = np.array((1200, 1200), dtype=np.int32)
PIECE_DISPLAY_SHAPE = np.array((800, 1067), dtype=np.int32)

CALIBRATION_INPUT_IMAGE_SHAPE = np.array((160, 160), dtype=np.int32)
PIECE_INPUT_IMAGE_SHAPE = np.array((140, 80), dtype=np.int32)

CALIBRATION_PREPROCESSING_NORMALIZATION = {"mean": 0.456, "std": 0.224}
PIECE_PREPROCESSING_NORMALIZATION = {"mean": 0.456, "std": 0.224}

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

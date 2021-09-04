""" Tool for labelling chessboard photos with camera calibration information.
"""

import argparse
import os

import cv2 as cv
import numpy as np
from rookowl.calibration import (CHESSBOARD_LABEL_POINTS,
                                 calibrate_from_points, draw_chessboard_grid)
from rookowl.global_var import CALIBRATION_DISPLAY_SHAPE
from rookowl.label import (get_photo_label_path, save_label,
                           search_unlabelled_photos)
from rookowl.utils.image import coord_from_px, coord_to_px

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class App:
    MAIN_WIN_NAME = "Chessboard Photo Labelling Tool"

    def __init__(self, dirpath):
        self.unlabelled_photo_paths = search_unlabelled_photos(dirpath)
        self.photo_path = None
        self.photo = None

    def run(self):
        self.make_window()
        self.load_next_photo()

        while self.photo is not None:
            self.update_view()

            key = cv.waitKey(0) & 0xff
            if key == 27:
                break

            if key == 32:
                self.accept_label()

        cv.destroyAllWindows()

    def move_envelope_point(self, point_index, xy):
        self.envelope_coords[point_index] = (float(xy[0]), float(xy[1]))

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == 2):
            best_index = None
            best_distance = None

            for index, coord in enumerate(self.envelope_coords):
                ex, ey = coord_to_px(coord)
                distance = (x - ex)**2 + (y - ey)**2
                if best_distance is None or distance < best_distance:
                    best_index = index
                    best_distance = distance

            self.move_envelope_point(
                best_index, coord_from_px((x, y)))

            self.update_view()

    def make_window(self):
        cv.namedWindow(self.MAIN_WIN_NAME)
        cv.setMouseCallback(self.MAIN_WIN_NAME, self.on_mouse_click)

    def update_view(self):
        image = self.photo.copy()
        self.draw_envelope(image)
        cv.imshow(self.MAIN_WIN_NAME, image)

    def draw_envelope(self, image):
        calibration, _ = calibrate_from_points(
            CHESSBOARD_LABEL_POINTS, self.envelope_coords)
        draw_chessboard_grid(image, calibration)

        for coord in self.envelope_coords:
            cv.circle(image, coord_to_px(coord), radius=8,
                      color=(255, 255, 255), thickness=2)

    def load_photo_image(self, filepath):
        photo = cv.imread(filepath)
        image = cv.resize(photo, (CALIBRATION_DISPLAY_SHAPE[1], CALIBRATION_DISPLAY_SHAPE[0]),
                          interpolation=cv.INTER_LANCZOS4)
        return image

    def load_next_photo(self):
        if not self.unlabelled_photo_paths:
            self.photo_path = None
            self.photo = None
            return False

        self.photo_path = self.unlabelled_photo_paths[0]
        del self.unlabelled_photo_paths[0]

        print(f"Loading image: {self.photo_path}")

        self.photo = self.load_photo_image(self.photo_path)

        self.reset_envelope()
        return True

    def reset_envelope(self):
        self.envelope_coords = np.array([
            (0.0, 0.0),

            (-2/8, +2/8),
            (+2/8, +2/8),
            (+2/8, -2/8),
            (-2/8, -2/8),

            (-0.5, 0.5),
            (0.5, 0.5),
            (0.5, -0.5),
            (-0.5, -0.5),
        ], dtype=np.float32)

    def accept_label(self):
        calibration, rms_error = calibrate_from_points(
            CHESSBOARD_LABEL_POINTS, self.envelope_coords)

        label = {
            "image_filename": os.path.basename(self.photo_path),
            "object_points": CHESSBOARD_LABEL_POINTS,
            "image_points": self.envelope_coords,
            "rms_error": rms_error,
            "calibration": calibration
        }

        label_path = get_photo_label_path(self.photo_path)
        save_label(label, path=label_path, copylabel=False)
        print(f"Label saved to: {label_path}")

        self.load_next_photo()


# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", dest="dataset_dir", type=str, default="dataset",
                        help="dataset directory path, where chessboard photographs are placed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = App(args.dataset_dir)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

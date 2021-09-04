""" Functions for searching, loading and saving labels from the dataset directory.
"""

import copy
import json
import os
import random

import numpy as np

from rookowl.calibration import Calibration
from rookowl.global_var import IMAGE_EXTENSIONS

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def get_photo_label_path(filepath):
    return ".".join(os.path.splitext(filepath)[0:-1]) + ".label.json"


def search_unlabelled_photos(dirpath, extensions=IMAGE_EXTENSIONS, verbose=True, shuffle=True):
    extensions = [f".{ext}" for ext in extensions]
    photo_count = 0
    paths = []
    for path in os.listdir(dirpath):
        rel_path = os.path.join(dirpath, path)
        if not os.path.isfile(rel_path) or os.path.splitext(path)[-1] not in extensions:
            continue
        photo_count += 1
        if os.path.isfile(get_photo_label_path(rel_path)):
            continue
        paths.append(rel_path)
    if verbose:
        print(
            f"Found {photo_count} images out of which {len(paths)} are unlabelled.")
    if shuffle:
        random.shuffle(paths)
    return paths


def search_stockphotos(dirpath, extensions=IMAGE_EXTENSIONS):
    extensions = [f".{ext}" for ext in extensions]
    paths = []
    for path in os.listdir(dirpath):
        rel_path = os.path.join(dirpath, path)
        if not os.path.isfile(rel_path) or os.path.splitext(path)[-1] not in extensions:
            continue
        paths.append(rel_path)
    return paths


def load_labels(dirpath, with_state_only=False, with_pieces_only=False):
    labels = []
    for path in os.listdir(dirpath):
        rel_path = os.path.join(dirpath, path)
        if not os.path.isfile(rel_path) or not path.endswith(".label.json"):
            continue

        with open(rel_path, "r") as file:
            label = json.loads(file.read())

        if with_state_only:
            if "state" not in label:
                continue
            if with_pieces_only and label["state"] == 64 * ".":
                continue

        label["source_path"] = rel_path
        label["image_filepath"] = os.path.join(
            dirpath, label["image_filename"])
        label["calibration"] = Calibration(
            np.array(label.pop("rvec"), dtype=np.float32),
            np.array(label.pop("tvec"), dtype=np.float32),
            np.array(label.pop("camera_matrix"), dtype=np.float32),
            np.array(label.pop("dist_coeffs"), dtype=np.float32))
        label["image_points"] = np.array(
            label.pop("image_points"), dtype=np.float32)
        label["object_points"] = np.array(
            label.pop("object_points"), dtype=np.float32)

        labels.append(label)

    labels.sort(key=lambda x: x["image_filepath"])
    return labels


def save_label(label, path=None, copylabel=True, writefile=True):
    if copylabel:
        label = copy.deepcopy(label)

    if path is None:
        assert "source_path" in label, "'path' not specified: know not where to save the label to"
        path = label["source_path"]
    label.pop("source_path", None)
    label.pop("image_filepath", None)
    assert "image_filename" in label

    assert isinstance(label["object_points"], np.ndarray)
    label["object_points"] = label["object_points"].tolist()
    assert isinstance(label["image_points"], np.ndarray)
    label["image_points"] = label["image_points"].tolist()
    assert "rms_error" in label

    calibration = label.pop("calibration")
    label["rvec"] = calibration.rvec.tolist()
    label["tvec"] = calibration.tvec.tolist()
    label["camera_matrix"] = calibration.camera_matrix.tolist()
    label["dist_coeffs"] = calibration.dist_coeffs.tolist()

    content = json.dumps(label, indent=4)

    if writefile:
        with open(path, "w") as file:
            file.write(content)

    return content

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

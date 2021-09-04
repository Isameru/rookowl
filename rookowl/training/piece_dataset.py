""" Contains `PieceDataset`, which is a PyTorch dataset providing images for piece recognition training and evaluation.
"""

import random

import numpy as np
import torch
from rookowl.global_var import (PIECE_INPUT_IMAGE_SHAPE,
                                PIECE_PREPROCESSING_NORMALIZATION)
from rookowl.label import load_labels
from rookowl.piece import PIECE_CODE_MAP, PIECE_TEXT_MAP
from rookowl.piece_image import segment_pieces_from_label
from torch.utils.data import Dataset
from torchvision import transforms

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class PieceDataset(Dataset):
    """ PyTorch dataset providing images for piece recognition training and evaluation.
    """

    def __init__(self, label_dir: str, shuffle, augment: bool = False, attach_preview: bool = False, with_pieces_only: bool = False):
        self.labels = load_labels(
            label_dir, with_state_only=True, with_pieces_only=with_pieces_only)
        self.length = len(self.labels) * 64
        self.shuffle = shuffle
        self.augment = augment
        self.attach_preview = attach_preview
        self.labels_order = list(range(len(self.labels)))
        self.piece_order = list(range(64))
        if shuffle:
            random.shuffle(self.labels_order)
        self.loaded_label_index = None
        self.loaded_label = None
        self.loaded_piece_images = None

    def prepare_label(self, label_index: int):
        if label_index != self.loaded_label_index:
            self.loaded_label_index = label_index
            self.loaded_label = self.labels[label_index]
            self.loaded_piece_images = segment_pieces_from_label(
                self.loaded_label, self.augment)
            if self.shuffle:
                random.shuffle(self.piece_order)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label_index = self.labels_order[index // 64]
        self.prepare_label(label_index)
        piece_index = self.piece_order[index % 64]
        piece_image = self.loaded_piece_images[piece_index]
        piece_symbol = self.loaded_label["state"][piece_index]

        if self.attach_preview:
            image_preview = piece_image

        piece_image = self.preprocess(piece_image)

        piece = torch.tensor(
            PIECE_CODE_MAP[piece_symbol], dtype=torch.int64)

        sample = {
            "image": piece_image,
            "piece": piece
        }

        if self.attach_preview:
            sample["image_preview"] = image_preview
            sample["piece_text"] = PIECE_TEXT_MAP[piece_symbol]

        return sample

    def preprocess(self, image):
        assert image.dtype == np.uint8
        assert all(image.shape == PIECE_INPUT_IMAGE_SHAPE)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.type(torch.float32) / 255.0
        image = transforms.Normalize(
            **PIECE_PREPROCESSING_NORMALIZATION)(image)
        return image

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

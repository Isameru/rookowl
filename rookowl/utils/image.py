""" Image-related helper utilities used widely in the project.
"""

import random

import cv2 as cv
import numpy as np

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def coord_to_px(shape, xy):
    assert len(shape) == 2 or (len(shape) == 3 and shape[2] <= 3)
    height, width = shape[0:2]
    return (int(((1.0 + xy[0]) * width) // 2),
            int(((1.0 - xy[1]) * height) // 2))


def coord_vec_to_px(shape, xy_vec):
    assert len(shape) == 2 or (len(shape) == 3 and shape[2] <= 3)
    height, width = shape[0:2]
    return [
        (int(((1.0 + xy[0]) * width) // 2),
         int(((1.0 - xy[1]) * height) // 2))
        for xy in xy_vec]


def coord_from_px(shape, xy):
    assert len(shape) == 2 or (len(shape) == 3 and shape[2] <= 3)
    height, width = shape[0:2]
    x, y = xy
    return np.array((
        2.0 * (x / (width - 1)) - 1.0,
        1.0 - 2.0 * (y / (height - 1))
    ), dtype=np.float32)


def coord_vec_from_px(shape, xy_vec):
    assert len(shape) == 2 or (len(shape) == 3 and shape[2] <= 3)
    x_factor = 1 / (shape[1] - 1)
    y_factor = 1 / (shape[0] - 1)
    return np.array([(
        2.0 * (xy[0] * x_factor) - 1.0,
        1.0 - 2.0 * (xy[1] * y_factor)
    ) for xy in xy_vec], dtype=np.float32)


# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def generate_contour_mask(shape, points_px, color=(1,)):
    """ Draws a black and white mask of a given `shape`.
        Effectively, one-bit polygon made of vertices `points_px` is drawn on a black canvas.
    """
    assert len(points_px) == 4
    mask = np.zeros(shape, dtype=np.uint8)
    cv.drawContours(mask, [np.array(points_px, dtype=np.int32)
                           ], -1, color, -1, cv.LINE_4)
    return mask


def noise_image(image, max_noise_value: int):
    assert image.dtype == np.uint8
    assert max_noise_value >= 0 and max_noise_value < 256
    if max_noise_value > 0:
        noise = np.zeros(image.shape, np.int8)
        cv.randn(noise, 0, max_noise_value)
        image = cv.add(image, noise, dtype=cv.CV_8UC3)
    return image


def rotate_image(image, image_points_px, center_point_px, angle, border_color=cv.BORDER_CONSTANT):
    rotation_matrix = cv.getRotationMatrix2D(
        tuple(center_point_px), angle, 1.0)
    image = cv.warpAffine(image, rotation_matrix,
                          (image.shape[1], image.shape[0]), borderValue=border_color)

    def rotate_coord(c):
        return cv.transform(np.array([[c]], dtype=np.float32), rotation_matrix).squeeze()
    image_points_px = [rotate_coord(p) for p in image_points_px]

    return image, image_points_px


def crop_image(image, image_points_px, crop_rect_px, border_color=None):
    image_height, image_width = image.shape[0:2]
    crop_x, crop_y, crop_w, crop_h = crop_rect_px
    crop_x2 = crop_x + crop_w
    crop_y2 = crop_y + crop_h

    left_abroad = 0 if crop_x >= 0 else -crop_x
    right_abroad = 0 if crop_x2 <= image_width else crop_x2 - image_width
    top_abroad = 0 if crop_y >= 0 else - crop_y
    bottom_abroad = 0 if crop_y2 <= image_height else crop_y2 - image_height

    if left_abroad == 0 and right_abroad == 0 and top_abroad == 0 and bottom_abroad == 0:
        image = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    else:
        assert border_color is not None
        new_image = np.full((crop_h, crop_w), border_color, image.dtype)

        new_image[top_abroad:crop_h-top_abroad-bottom_abroad, left_abroad:crop_w-left_abroad-right_abroad] = \
            image[crop_y+top_abroad:crop_y+crop_h-top_abroad-bottom_abroad,
                  crop_x+left_abroad:crop_x+crop_w-left_abroad-right_abroad]

        image = new_image

    def crop_coord(c):
        return (c[0] - crop_x, c[1] - crop_y)
    image_points_px = [crop_coord(p) for p in image_points_px]

    return image, image_points_px


def paste_image(dst_image, src_image, src_image_points_px, offset):
    x1 = max(0, offset[0])
    x2 = min(dst_image.shape[1], offset[0] + src_image.shape[1])
    y1 = max(0, offset[1])
    y2 = min(dst_image.shape[0], offset[1] + src_image.shape[0])

    if x1 < x2 and y1 < y2:
        dst_image[y1:y2, x1:x2] = src_image[y1-offset[1]:y2-offset[1], x1-offset[0]:x2-offset[0]]

    def offset_coord(c):
        return (c[0] + offset[0], c[1] + offset[1])

    dst_image_points_px = [offset_coord(p) for p in src_image_points_px]
    return dst_image_points_px


def scale_image(image, image_points_px, dst_shape, interpolation=cv.INTER_LINEAR):
    src_h, src_w = image.shape[0:2]
    dst_h, dst_w = dst_shape
    image = cv.resize(image, (dst_w, dst_h), interpolation=interpolation)

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    def scale_coord(c):
        return (c[0] * scale_x, c[1] * scale_y)
    image_points_px = [scale_coord(p) for p in image_points_px]

    return image, image_points_px


def draw_random_blobs(image, max_blob_count, blob_probability, blob_size_divider):
    for _ in range(max_blob_count):
        if random.uniform(0.0, 1.0) > blob_probability:
            continue
        center = (random.randint(0, image.shape[1]), random.randint(
            0, image.shape[0]))
        axes = (random.randint(
            10, image.shape[1] // blob_size_divider), random.randint(10, image.shape[0] // blob_size_divider))
        fill_color = random.randint(0, 255)
        poly_points = cv.ellipse2Poly(center, axes, 0, 0, 360, 18)
        cv.fillPoly(image, np.array(
            [poly_points], dtype=np.int32), fill_color)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

""" Downloads beutiful random stock JPG photos from https://picsum.photos/ to a directory.
"""

import argparse
import os
import time
import urllib.request

from rookowl.global_var import DOWNLOAD_STOCKPHOTO_SHAPE

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", dest="stockphotos_dir", type=str, default="stockphotos",
                        help="stock photos directory path")
    parser.add_argument("-n", dest="total_num", type=int, default=1000,
                        help="desired total number of stock photos")
    parser.add_argument("-w", dest="wait", type=float, default=2.0,
                        help="wait interval between download requests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for i in range(args.total_num):
        filepath = os.path.join(args.stockphotos_dir, "{:05d}.jpg".format(i))
        if os.path.isfile(filepath):
            continue

        print(f"Downloading {filepath}...")
        urllib.request.urlretrieve(
            f"https://picsum.photos/{int(DOWNLOAD_STOCKPHOTO_SHAPE[1])}/{int(DOWNLOAD_STOCKPHOTO_SHAPE[0])}.jpg?grayscale&random", filepath)

        time.sleep(args.wait)  # Let's not be too greedy...

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

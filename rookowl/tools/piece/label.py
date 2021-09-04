""" Tool for extending the label information with the chess game state, i.e. the camera piece
    color and square content.
"""

import argparse

import cv2 as cv
from rookowl.global_var import PIECE_DISPLAY_SHAPE
from rookowl.label import load_labels, save_label

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_state_text(state_text: str, near_color: str):
    state = 64 * ["."]

    def put(piece: str, file: int, rank: int):
        assert piece in ["P", "p", "B", "b", "N",
                         "n", "R", "r", "Q", "q", "K", "k"]
        assert file >= 1 and file <= 8
        assert rank >= 1 and rank <= 8
        assert near_color in ["b", "w"]
        index = 8 * (rank-1) + (file-1)
        if near_color == "b":
            index = 63 - index
        state[index] = piece

    file, rank = 1, 8
    for c in state_text:
        if c == " ":
            break
        elif c == "/":
            file = 1
            rank -= 1
        elif c.isdigit():
            file += int(c)
        else:
            put(c, file, rank)
            file += 1

    return "".join(state)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class App:
    MAIN_WIN_NAME = "Piece Labelling Tool"

    def __init__(self, dirpath):
        self.labels = load_labels(dirpath)

    def run(self):
        self.make_window()

        for label in self.labels:
            if "state" in label:
                continue

            image = cv.imread(label["image_filepath"])
            image = cv.resize(
                image, (PIECE_DISPLAY_SHAPE[1], PIECE_DISPLAY_SHAPE[0]), interpolation=cv.INTER_CUBIC)
            cv.imshow(self.MAIN_WIN_NAME, image)
            cv.waitKey(1)

            print(80 * "_")
            print(f"Now presenting: {label['image_filename']}")
            near_color = input(
                "Which player holds the camera and it is a known state? [b/w/bs/ws/be/we] ")
            if not near_color in ["b", "w", "bs", "ws", "be", "we"]:
                print("Skipping to the next photo")
                continue

            if near_color in ["b", "w"]:
                state_text = input("What is this state? ")
            elif near_color == "bs":
                near_color = "b"
                state_text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            elif near_color == "ws":
                near_color = "w"
                state_text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            elif near_color == "be":
                near_color = "b"
                state_text = ""
            elif near_color == "we":
                near_color = "w"
                state_text = ""

            cv.waitKey(1)
            state = parse_state_text(state_text, near_color)
            print(f"Parsed state: '{state}'")

            label_path = label["source_path"]
            label["state"] = state
            label["side"] = near_color
            save_label(label, copylabel=False)
            print(f"Label '{label_path}' updated.")

            cv.waitKey(1)

        cv.destroyAllWindows()

    def make_window(self):
        cv.namedWindow(self.MAIN_WIN_NAME)

    def update_view():
        pass


# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument("-d", dest="dataset_dir", default="dataset",
                        help="dataset directory path, where chessboard photographs are placed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = App(args.dataset_dir)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

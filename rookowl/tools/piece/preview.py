""" Tool for previewing the piece images prepared for training or evaluating the piece recognition.
"""

import argparse

import cv2 as cv
import torch
from rookowl.piece import PIECE_CODE_COUNT, PIECE_TEXT_MAP
from rookowl.training.piece_dataset import PieceDataset

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


class App:
    MAIN_WIN_NAME = "Piece Recognition :: Training Images Preview Tool"

    def __init__(self, label_dir: str, model_file: str, purpose: str):
        assert purpose in ["train", "eval"]
        self.dataset = PieceDataset(
            label_dir, shuffle=(purpose == "train"), augment=(purpose == "train"), attach_preview=True)
        print(f"Dataset has {len(self.dataset)} piece images.")

        try:
            print(f"Loading model: {model_file}")
            self.model = torch.load(model_file)
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.model.eval()
        except Exception as ex:
            print(f"Failed to load the model: {str(ex)}")
            self.model = None

    def run(self):
        cv.namedWindow(self.MAIN_WIN_NAME)

        for sample_index in range(len(self.dataset)):
            sample = self.dataset[sample_index]
            cv.imshow(self.MAIN_WIN_NAME, sample["image_preview"])

            print(f"Presenting: {self.dataset.loaded_label['image_filepath']}")
            print(f"    Reference : {sample['piece_text']}")

            if self.model is not None:
                input_batch = sample["image"].unsqueeze(0)

                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')

                with torch.no_grad():
                    output = self.model(input_batch)[0]

                output = output.cpu().numpy()

                top = sorted([{"code": code, "value": output[code]}
                              for code in range(PIECE_CODE_COUNT)], key=lambda x: -x["value"])

                piece_texts = list(PIECE_TEXT_MAP.values())

                if top[0]["value"] <= 0:
                    print("    Failed to recognize the piece.")
                    print(f"    {output}")
                else:
                    print(
                        f"    Top #1    : {piece_texts[top[0]['code']]} ({top[0]['value']})")
                    top_2nd = top[1]
                    if top_2nd["value"] > 0:
                        print(
                            f"    Top #2    : {piece_texts[top_2nd['code']]} ({top_2nd['value']})")

            key = cv.waitKey(0) & 0xff
            if key == 27:
                break

        cv.destroyAllWindows()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", type=str, help="train, eval")
    parser.add_argument("-d", dest="dataset_dir", default="dataset/validation",
                        help="dataset directory path, where chessboard images and their labels are placed")
    parser.add_argument("-m", dest="model_file", type=str,
                        default="models/piece_model-best.pth", help="model file path to load")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = App(args.dataset_dir, args.model_file, args.purpose)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

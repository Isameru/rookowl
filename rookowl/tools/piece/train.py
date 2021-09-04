""" Trains the piece recognition model.
"""

import argparse
import gc
import math
import os
import time

import torch
from rookowl.training.piece_dataset import PieceDataset

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

EPOCHS = 10000
SAVE_PER_EPOCHS = 1
BATCH_SIZE = 22
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 0.00001

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def free_unused_space():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class App:
    def __init__(self, model_file, label_dir):
        self.model_path = model_file
        self.dataset = PieceDataset(
            os.path.join(label_dir, "training"), shuffle=True, augment=True, attach_preview=False, with_pieces_only=True)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        self.eval_dataset = PieceDataset(
            os.path.join(label_dir, "validation"), shuffle=False, augment=False, attach_preview=False, with_pieces_only=True)
        self.eval_dataset_loader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        print(f"Loading model: {model_file}")
        self.model = torch.load(model_file)

        for name, param in self.model.named_parameters():
            param.requires_grad = True

        params_to_update = []
        for name, param in self.model.named_parameters():
            if True:
                params_to_update.append(param)
                print(f"    {name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params_to_update, lr=LEARNING_RATE)

    def train_sample(self, sample):
        self.optimizer.zero_grad()

        input = sample["image"].to(self.device)
        exp_output = sample["piece"].to(self.device)

        output = self.model(input)

        input.detach()
        del input

        loss = self.criterion(output, exp_output)

        output.detach()
        del output
        exp_output.detach()
        del exp_output

        if not math.isfinite(loss.item()):
            print("error: Training loss diverged.")
            exit(-1)

        free_unused_space()

        loss.backward()
        self.optimizer.step()

        loss.detach()

        free_unused_space()
        return loss

    def eval_sample(self, sample):
        input = sample["image"].to(self.device)
        exp_output = sample["piece"].to(self.device)

        with torch.no_grad():
            output = self.model(input)

            input.detach()
            del input

            loss = self.criterion(output, exp_output)

        output.detach()
        del output
        exp_output.detach()
        del exp_output

        if not math.isfinite(loss.item()):
            print("error: Evaluation loss diverged.")
            exit(-1)

        free_unused_space()
        return loss.detach()

    def eval_model(self, verbose=True):
        free_unused_space()
        if verbose:
            print("Evaluating the model...")
        self.model.eval()
        loss = 0
        for sample in self.eval_dataset_loader:
            loss += (sample["image"].shape[0] / EVAL_BATCH_SIZE) * \
                self.eval_sample(sample).item()
        if verbose:
            print(
                f"Evaluation loss: {type(self.criterion).__name__} = {loss:.9f}")
        return loss

    def run(self):
        step = 0
        best_loss = self.eval_model()
        for epoch in range(EPOCHS):
            print(f"Starting epoch: {epoch}")
            self.model.train()
            for sample in self.dataset_loader:
                loss = self.train_sample(sample)
                print(
                    f"Step {step}: {type(self.criterion).__name__} = {loss.item():.9f}")
                step += 1

            if (epoch + 1) % SAVE_PER_EPOCHS == 0:
                loss = self.eval_model()

                print("Saving model... (do not interrupt)")
                time.sleep(1)
                torch.save(self.model, self.model_path)
                print("Done.")

                if loss < best_loss:
                    print("Saving best model... (do not interrupt)")
                    time.sleep(1)
                    loss_text = f"{loss:.9f}".replace(".", "_")
                    torch.save(
                        self.model, f"{self.model_path}-loss_{loss_text}")
                    print("Done.")
                    best_loss = loss


# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument("-m", dest="model_file", default="models/piece_model.pth",
                        help="model file path to load, train and save")
    parser.add_argument("-d", dest="dataset_dir", default="dataset",
                        help="dataset directory path, where chessboard images and their labels are placed")
    args = parser.parse_args()

    app = App(args.model_file, args.dataset_dir)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

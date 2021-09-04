""" Trains the calibration model.
"""

import argparse
import gc
import math
import os

import torch
from rookowl.training.calibration_dataset import CalibrationDataset
from rookowl.training.calibration_model import CalibrationModel

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

EPOCHS = 10000
SAVE_PER_EPOCHS = 5
BATCH_SIZE = 16
ACCUM_STEPS = 4
EVAL_BATCH_SIZE = 2 * BATCH_SIZE
LEARNING_RATE = 0.000001

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def free_unused_space():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class App:
    def __init__(self, model_file, label_dir, stockphotos_dir):
        self.model_path = model_file
        self.dataset = CalibrationDataset(
            os.path.join(label_dir, "training"), augment=True, stockphotos_dir=stockphotos_dir, preview=False)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

        self.eval_dataset = CalibrationDataset(
            os.path.join(label_dir, "validation"), augment=False, preview=False)
        self.eval_dataset_loader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

        print(f"Loading model: {model_file}")
        self.model = CalibrationModel()
        self.model.load_state_dict(torch.load(model_file), strict=False)

        for name, param in self.model.named_parameters():
            param.requires_grad = True

        params_to_update = []
        for name, param in self.model.named_parameters():
            if True:  # name.startswith("features.0."):
                params_to_update.append(param)
                print(f"    {name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE)

    def train_sample(self, sample):
        free_unused_space()
        # self.optimizer.zero_grad()

        input = sample["image"].to(self.device)
        exp_output = sample["image_points"].to(self.device)

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
        # self.optimizer.step()

        loss.detach()

        free_unused_space()
        return loss

    def eval_sample(self, sample):
        free_unused_space()
        input = sample["image"].to(self.device)
        exp_output = sample["image_points"].to(self.device)

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
            for sample_index, sample in enumerate(self.dataset_loader):
                if step % ACCUM_STEPS == 0:
                    print("Clearing gradients.")
                    self.optimizer.zero_grad()
                loss = self.train_sample(sample)
                print(
                    f"Step {step}: {type(self.criterion).__name__} = {loss.item():.9f}")
                step += 1
                if step % ACCUM_STEPS == 0:
                    print("Optimizing using the accumulated gradients.")
                    self.optimizer.step()

            if (epoch + 1) % SAVE_PER_EPOCHS == 0:
                loss = self.eval_model()

                print("Saving model... (do not interrupt)")
                torch.save(self.model.state_dict(), self.model_path)
                print("Done.")

                if loss < best_loss:
                    print("Saving best model... (do not interrupt)")
                    loss_text = f"{loss:.9f}".replace(".", "_")
                    torch.save(self.model.state_dict(),
                               f"{self.model_path}-loss_{loss_text}")
                    print("Done.")

                    best_loss = loss

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", dest="model_file", default="models/calibration_model.pth",
                        help="model file path to load, train and save")
    parser.add_argument("-d", dest="dataset_dir", default="dataset",
                        help="dataset directory path, where chessboard images and their labels are placed")
    parser.add_argument("-s", dest="stockphotos_dir", default="stockphotos",
                        help="stock photos directory path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = App(args.model_file, args.dataset_dir, args.stockphotos_dir)
    app.run()

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

# This python file contains helper classes and functions which make the latter notebooks
# more readable.
import re
import importlib

import numpy as np
import torch
import torchinfo
from mnist1d.data import get_dataset_args, make_dataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt, colormaps


class MNIST1D(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        test_size: float = 0.1,
        mnist1d_args: dict = get_dataset_args(),
        seed: int = 42,
    ):
        super().__init__()

        self.is_training = train
        self.data = make_dataset(mnist1d_args)

        # dataset split, the same can be done with torch.utils.data.random_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.data["x"],
            self.data["y"],
            test_size=test_size,
            random_state=seed,
        )

        # normalize the data
        self.X_loc = np.min(X_train)
        self.X_scale = np.max(X_train) - np.min(X_test)

        # decide training and testing
        if train:
            self.X = (X_train - self.X_loc) / self.X_scale
            self.y = y_train
        else:
            # use the same normalisation strategy as during training
            self.X = (X_test - self.X_loc) / self.X_scale
            self.y = y_test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        X = torch.from_numpy(self.X[index : index + 1, ...].astype(np.float32))
        y = torch.from_numpy(self.y[index, ...].astype(np.int64))

        return X, y


def count_params(torch_model: torch.nn.Module):
    """
    function to calculate the number of trainable parameters of a torch.nn.Module

    Parameters
    ----------
    torch_model : torch.nn.Module
        torch model to calculate number of parameters for


    Examples
    --------
    > testmodel = torch.nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3, padding=1, bias=False)
    > count_params(testmodel)
    3

    """

    value = sum([p.view(-1).shape[0] for p in torch_model.parameters()])

    return value


def model_summary(*args, **kwds):
    return torchinfo.summary(
        *args, col_names=("input_size", "output_size", "num_params"), **kwds
    )


def import_check(fn: str):
    name_map = {"scikit-learn": "sklearn"}
    with open(fn, "r") as fd:
        for line in fd.readlines():
            m = re.match(r"^([^#]\s*[a-z0-9_\-]+).*$", line.strip())
            if m is not None:
                pkg_name = name_map.get(m.group(1), m.group(1))
                try:
                    pkg = importlib.import_module(pkg_name)
                    if hasattr(pkg, "__version__"):
                        vv = f"version {pkg.__version__}"
                    else:
                        vv = "(no version info)"
                    print(f"{pkg_name:20}: found, {vv:25}")
                except ImportError:
                    print(f"{pkg_name}: NOT FOUND!")


# https://matplotlib.org/stable/gallery/color/color_cycle_default.html
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors_10 = prop_cycle.by_key()["color"]
assert len(colors_10) == 10


def get_label_colors(y, cmap=colormaps.get_cmap("jet")):
    n_unique_labels = len(np.unique(y))
    if n_unique_labels == 10:
        colors = colors_10
    else:
        rng = np.random.default_rng(seed=42)
        colors = [cmap(rng.random()) for _ in range(n_unique_labels)]
    return [colors[y_i] for y_i in y]

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Autoencoders

An autoencoder is a type of artificial neural network used for learning
efficient encodings of input data. It's essentially a network that attempts to
replicate its input (encoding) as its output (decoding), but the network is
designed in such a way that it must learn an efficient representation
(compression) for the input data in order to map it back to itself.

The importance of autoencoders lies in their ability to learn the underlying
structure of complex data, making them valuable tools for scientific data
analysis. Here's how:

1. Dimensionality Reduction: Autoencoders can be used to reduce the
dimensionality of high-dimensional data while preserving its essential
characteristics. This is particularly useful in cases where the high
dimensionality makes computations slow or the data overfitting occurs.

2. Denoising: By training autoencoders on noisy versions of the data, they can
learn to remove noise from the original data, making it cleaner and easier to
analyze.

3. Anomaly Detection: The encoder part of the autoencoder can be used to
represent the input data in a lower-dimensional space. Any data point that is
far from the rest in this space can be considered an anomaly, as it doesn't fit
the pattern learned by the autoencoder during training.

4. Generative Modeling: Autoencoders can be used as generative models, allowing
them to generate new data that are similar to the original data. This can be
useful in various scientific applications, such as creating synthetic data or
for exploring the data space.

5. Feature Learning: Autoencoders can learn useful features from raw data,
which can then be used as inputs for other machine learning models, improving
their performance.

In summary, autoencoders are a powerful tool for scientific data analysis due
to their ability to learn the underlying structure of complex data.
"""

# %% [markdown]
"""
## An autoencoder for denoising

In the next cells, we will face a situation in which the quality of the data is
rather poor. There is a lot of noise added to the dataset which is hard to
handle. We will set up an autoencoder to tackle the task of **denoising**, i.e.
to remove stochastic fluctuations from the input as best as possible.

First, let's prepare a dataset which contains a noisy signal that we wish to
denoise. For that we use the MNIST1D dataset from earlier and add artificial
noise. The autoencoder's task is to remove this noise by learning the
characteristics of the data.
"""

# %%
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import gridspec, pyplot as plt

from mnist1d.data import get_dataset_args, make_dataset

from utils import model_summary, MNIST1D

np.random.seed(13)
torch.random.manual_seed(12)

# %%
# disable noise for a clear reference
clean_config = get_dataset_args()
clean_config.iid_noise_scale = 0
clean_config.corr_noise_scale = 0
clean_config.seed = 40
clean = make_dataset(clean_config)
cleanX, cleany = clean["x"], clean["y"]

# use iid noise only for the time being
noisy_config = get_dataset_args()
noisy_config.iid_noise_scale = 0.05
noisy_config.corr_noise_scale = 0
noisy_config.seed = 40
data = make_dataset(noisy_config)

X, y = data["x"], data["y"]

# 4000 data points of dimension 40
print(f"{X.shape=}")
print(f"{y.shape=}")

# The MNIST1D Dataset produces numpy arrays. Lets convert to torch.Tensor, with
# torch.float32 dtype.
X = torch.from_numpy(X).float()

# %% [markdown]
"""
Now, let's plot the data which we would like to use.
"""

# %%
fig, ax = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)
color_noisy = "tab:blue"
color_clean = "tab:orange"

for sample in range(10):
    col = sample % 5
    row = sample // 5
    ax[row, col].plot(X[sample, ...], label="noisy", color=color_noisy)
    ax[row, col].plot(cleanX[sample, ...], label="clean", color=color_clean)
    label = y[sample]
    ax[row, col].set_title(f"label {label}")
    if row == 1:
        ax[row, col].set_xlabel("samples / a.u.")
    if col == 0:
        ax[row, col].set_ylabel("intensity / a.u.")
    if col == 4 and row == 0:
        ax[row, col].legend()

fig.suptitle("MNIST1D examples")
fig.savefig("mnist1d_noisy_first10.svg")

# %% [markdown]
"""
As we can see, the data is filled with jitter. Furthermore, it is interesting
to note that our dataset is still far from trivial. Have a look at all signals
which are assigned to a certain label. Could you detect them?

## Designing an autoencoder

The [autoencoder architecture](https://en.wikipedia.org/wiki/Autoencoder) is
well illustrated on Wikipedia. We reproduce [the
image](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png) by
[Michaela
Massi](https://commons.wikimedia.org/w/index.php?title=User:Michela_Massi&action=edit&redlink=1)
here for convenience: <div style="display: block;margin-left:
auto;margin-right: auto;width: 75%;"><img
src="https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png"
alt="autoencoder schematic from Wikipedia by Michaela Massi, CC-BY 4.0"></div>

The architecture consists of three parts:

1. **the encoder** on the left: this small network ingests the input data `X`
   and compresses it into a smaller shape
2. the **code** in the center: this is the "bottleneck" which holds the
   **latent representation** `h`
3. **the decoder** on the right: reconstructs the output `X'` from the latent code `h`

The task of the autoencoder is to reconstruct the input as best as possible.
This task is far from easy, as the autoencoder is forced to shrink the data
into the latent space.

Since we have have the same MNIST1D data as before, we'll use convolutional
layers to build the autoencoder.
"""


# %%
class MyEncoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16):
        super().__init__()
        self.layers = torch.nn.Sequential()

        self.layers.append(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=nchannels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )

        for i in range(nlayers - 1):
            # convolve and shrink input width by 2x
            self.layers.append(
                torch.nn.Conv1d(
                    in_channels=nchannels,
                    out_channels=nchannels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            self.layers.append(torch.nn.ReLU())
            self.layers.append(
                torch.nn.Conv1d(
                    in_channels=nchannels,
                    out_channels=nchannels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )

        # convolve and keep input width
        self.layers.append(
            torch.nn.Conv1d(
                in_channels=nchannels, out_channels=1, kernel_size=3, padding=1
            )
        )

        # flatten and add a linear tail
        self.layers.append(torch.nn.Flatten())

    def forward(self, x):
        # convolutions in torch require an explicit channel dimension to be
        # present in the data in other words:
        # inputs of size (nbatch, 40) do not work,
        # inputs of size (nbatch, 1, 40) do work
        if len(x.shape) == 2:
            return self.layers(torch.unsqueeze(x, dim=1))
        else:
            return self.layers(x)


# %% [markdown]
"""
This decoder is not yet trained, so its weights are random. Still, lets apply
this to some input data and observe the input and output shapes. For this we'll
use the `model_summary()` helper function.
"""

# %%
enc = MyEncoder()

# extract only first 8 samples for testing
X_test = X[:8, ...]

latent_h = enc(X_test)

assert (
    latent_h.shape[-1] < X_test.shape[-1]
), f"{latent_h.shape[-1]} !< {X_test.shape[-1]}"

print(f"{X[:1, ...].shape=}")
model_summary(enc, input_size=X[:1, ...].shape)

# %% [markdown]
"""
The encoder takes a tensor of shape `[batch_size, 40]` (or `[batch_size, 1,
40]` with a channel dimension) and compresses that (with default settings for
`nlayers`) to a latent `h` of shape `[batch_size, 10]`. Above, we used
`batch_size=1` when calling `model_summary()`.

The encoder has been constructed. Now, we need to add a decoder object to
reconstruct from the latent space.
"""


# %%
class MyDecoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16):
        super().__init__()
        self.layers = torch.nn.Sequential()

        for i in range(nlayers - 1):
            inchannels = 1 if i == 0 else nchannels
            # deconvolve/upsample and grow input width by 2x
            self.layers.append(
                torch.nn.ConvTranspose1d(
                    in_channels=inchannels,
                    out_channels=nchannels,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    output_padding=1,
                )
            )
            self.layers.append(torch.nn.ReLU())
            self.layers.append(
                torch.nn.Conv1d(
                    in_channels=nchannels,
                    out_channels=nchannels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )

        # convolve and keep input width
        self.layers.append(
            torch.nn.Conv1d(
                in_channels=nchannels, out_channels=1, kernel_size=3, padding=1
            )
        )

        self.layers.append(torch.nn.Flatten())

    def forward(self, x):
        # convolutions in torch require an explicit channel dimension to be
        # present in the data in other words:
        # inputs of size (nbatch, 40) do not work,
        # inputs of size (nbatch, 1, 40) do work
        if len(x.shape) == 2:
            return self.layers(torch.unsqueeze(x, dim=1))
        else:
            return self.layers(x)


# %%
dec = MyDecoder()

X_prime = dec(latent_h)
assert (
    X_prime.squeeze(1).shape == X_test.shape
), f"{X_prime.squeeze(1).shape} != {X_test.shape}"

print(f"{latent_h[:1, ...].shape=}")
model_summary(dec, input_size=latent_h[:1, ...].shape)

# %% [markdown]
"""
We have now all the lego bricks in place to compose an autoencoder. We do this
by combining the encoder and decoder in yet another `torch.nn.Module`.
"""


# %%
class MyAutoencoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16):
        super().__init__()

        self.enc = MyEncoder(nlayers, nchannels)
        self.dec = MyDecoder(nlayers, nchannels)

    def forward(self, x):
        # construct the latents
        h = self.enc(x)

        # perform reconstruction
        x_prime = self.dec(h)

        return x_prime


# %% [markdown]
"""
We can test our autoencoder to make sure it works as expected similar to what we did above.
"""

# %%
model = MyAutoencoder()
X_prime = model(X_test)

assert (
    X_prime.squeeze(1).shape == X_test.shape
), f"{X_prime.squeeze(1).shape} != {X_test.shape}"

print(f"{X[:1, ...].shape=}")
model_summary(model, input_size=X[:1, ...].shape)


# %% [markdown]
"""
## Training an autoencoder

Training the autoencoder works the same as before.

1. create the dataset
2. create the loaders
3. setup the model
4. setup the optimizer
5. loop through epochs

Here we use a torch feature `StackDataset` to combine noisy inputs and clean
targets.
"""

# %%
# noisy data
dataset_train_noisy = MNIST1D(mnist1d_args=noisy_config, train=True)
dataset_test_noisy = MNIST1D(mnist1d_args=noisy_config, train=False)

# clean data
dataset_train_clean = MNIST1D(mnist1d_args=clean_config, train=True)
dataset_test_clean = MNIST1D(mnist1d_args=clean_config, train=False)

# stacked as paired sequences, like Python's zip()
dataset_train = torch.utils.data.StackDataset(
    dataset_train_noisy, dataset_train_clean
)
dataset_test = torch.utils.data.StackDataset(
    dataset_test_noisy, dataset_test_clean
)

batch_size = 64
train_dataloader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True
)
test_dataloader = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False
)

# %% [markdown]
#
# Let's inspect the data produced by the `DataLoader`. We look at the first batch
# of data, which we create by letting `train_dataloader` run one iteration.

# %%
train_noisy, train_clean = next(iter(train_dataloader))

X_train_noisy, y_train_noisy = train_noisy
X_train_clean, y_train_clean = train_clean

print(f"{len(train_noisy)=} {len(train_clean)=}")
print(f"{X_train_noisy.shape=} {y_train_noisy.shape=}")
print(f"{X_train_clean.shape=} {y_train_clean.shape=}")

# %% [markdown]
"""
We observe:

* The `DataLoader` (via the MNIST1D custom Dataset) has added a channel
  dimension, such that in each batch of `batch_size=64`, `X.shape` is `[64, 1,
  40]` rather than `[64, 40]`. That is just a convenience feature. Our model can
  handle either.
* The `DataLoader` also returns the labels `y_train_*` since that is part of the
  MNIST1D Dataset. We will discard them below, since for training an
  autoencoder, we only need the inputs `X`.
"""

# %% [markdown]
"""
Now lets iterate through the data and verify that we combined the correct noisy
and clean data points using StackDataset. We will look at the first `nrows *
ncols` batches. For each batch, we plot noisy and clean data for a randomly
picked data point `idx_in_batch`, which can be any number between 0 and
`batch_size - 1`.
"""

# %%
grid = gridspec.GridSpec(nrows=3, ncols=4)
fig = plt.figure(figsize=(5 * grid.ncols, 5 * grid.nrows))

for batch_idx, (gs, (train_noisy, train_clean)) in enumerate(
    zip(grid, train_dataloader)
):
    X_train_noisy, y_train_noisy = train_noisy
    X_train_clean, y_train_clean = train_clean
    assert (y_train_noisy == y_train_clean).all()
    ax = fig.add_subplot(gs)
    idx_in_batch = np.random.randint(0, len(y_train_noisy))
    ax.plot(
        X_train_noisy[idx_in_batch].squeeze(), label="noisy", color=color_noisy
    )
    ax.plot(
        X_train_clean[idx_in_batch].squeeze(), label="clean", color=color_clean
    )
    title = "\n".join(
        (
            f"batch={batch_idx+1} {idx_in_batch=}",
            f"labels: noisy={y_train_noisy[idx_in_batch]} clean={y_train_clean[idx_in_batch]}",
        )
    )
    ax.set_title(title)
    ax.legend()

# %% [markdown]
"""
Let's define a helper function that will run the training.
"""

# %%
def train_autoencoder(
    model,
    opt,
    crit,
    train_dataloader,
    test_dataloader,
    max_epochs,
    log_every=5,
    use_gpu=False,
):
    results = {"train_losses": [], "test_losses": []}
    ntrainsteps = len(train_dataloader)
    nteststeps = len(test_dataloader)
    train_loss, test_loss = (
        torch.empty((ntrainsteps,)),
        torch.empty((nteststeps,)),
    )

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.train()

    for epoch in range(max_epochs):
        # perform train for one epoch
        for idx, (train_noisy, train_clean) in enumerate(train_dataloader):
            # Discard labels if using StackDataset
            if isinstance(train_noisy, Sequence):
                X_train_noisy = train_noisy[0]
                X_train_clean = train_clean[0]
            else:
                X_train_noisy = train_noisy
                X_train_clean = train_clean

            # forward pass
            X_prime_train = model(X_train_noisy.to(device))

            # compute loss
            loss = crit(X_prime_train, X_train_clean.squeeze().to(device))

            # compute gradient
            loss.backward()

            # apply weight update rule
            opt.step()

            # set gradients to 0
            opt.zero_grad()

            train_loss[idx] = loss.item()

        for idx, (test_noisy, test_clean) in enumerate(test_dataloader):
            # Discard labels if using StackDataset
            if isinstance(test_noisy, Sequence):
                X_test_noisy = test_noisy[0]
                X_test_clean = test_clean[0]
            else:
                X_test_noisy = test_noisy
                X_test_clean = test_clean

            X_prime_test = model(X_test_noisy.to(device))
            loss_ = crit(X_prime_test, X_test_clean.squeeze().to(device))
            test_loss[idx] = loss_.item()

        results["train_losses"].append(train_loss.mean())
        results["test_losses"].append(test_loss.mean())

        if (epoch + 1) % log_every == 0 or (epoch + 1) == max_epochs:
            print(
                f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.4f}; test loss {test_loss.mean():03.4f}"
            )
    return results

# %% [markdown]
"""
Now we define the autoencoder model, the optimizer, the loss function, as well
as hyper-parameters such as the optimizer step size (`learning_rate`).

Again, we inspect the model using `model_summary()`. This time, we use an input
tensor from `train_dataloader`, which has shape `[batch_size, 1, 40]`.
"""

# %%
nsamples = len(dataset_train_noisy) + len(dataset_test_noisy)
assert (
    nsamples == 4_000
), f"number of samples for MNIST1D is not 4_000 but {nsamples}"

model = MyAutoencoder(nchannels=32)
print(
    model_summary(model, input_size=next(iter(train_dataloader))[0][0].shape)
)

learning_rate = 1e-3
max_epochs = 20
log_every = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # our loss function

# %% [markdown]
"""
Run training.

Note that if you re-execute this cell with*out* reinstantiating `model` above,
you will continue training with the so-far best model as start point.
"""

# %%
results = train_autoencoder(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    max_epochs,
    log_every,
)

# %% [markdown]
"""
# Plot loss (train progress) and predictions
"""

# %%

# Move model to CPU (only if a GPU was used, else this does nothing) and put in
# eval mode.
model = model.cpu()
model.eval()


fig, ax = plt.subplots()
ax.plot(results["train_losses"], color="b", label="train")
ax.plot(results["test_losses"], color="orange", label="test")
ax.set_xlabel("epoch")
ax.set_ylabel("average MSE Loss / a.u.")
ax.set_yscale("log")
ax.set_title("Loss")
ax.legend()
fig.savefig("mnist1d_noisy_conv_autoencoder_loss.svg")


with torch.no_grad():
    grid = gridspec.GridSpec(nrows=3, ncols=4)
    fig = plt.figure(figsize=(5 * grid.ncols, 5 * grid.nrows))

    for batch_idx, (gs, (test_noisy, test_clean)) in enumerate(
        zip(grid, test_dataloader)
    ):
        X_test_noisy, y_test_noisy = test_noisy
        X_test_clean, y_test_clean = test_clean
        assert (y_test_noisy == y_test_clean).all()
        ax = fig.add_subplot(gs)
        idx_in_batch = np.random.randint(0, len(y_test_noisy))
        ax.plot(
            X_test_noisy[idx_in_batch].squeeze(),
            label="noisy",
            color=color_noisy,
        )
        ax.plot(
            X_test_clean[idx_in_batch].squeeze(),
            label="clean",
            color=color_clean,
        )
        ax.plot(
            model(X_test_noisy[idx_in_batch]).squeeze(),
            label="prediction",
            color="tab:red",
            lw=2,
        )
        title = "\n".join(
            (
                f"batch={batch_idx+1} {idx_in_batch=}",
                f"labels: noisy={y_test_noisy[idx_in_batch]} clean={y_test_clean[idx_in_batch]}",
            )
        )
        ax.set_title(title)
        ax.legend()

fig.savefig("mnist1d_noisy_conv_autoencoder_predictions.svg")


# %% [markdown]
"""
We can see that the autoencoder smoothed the input signal when producing a
reconstruction. This denoising effect can be quite helpful in practice. The
core reasons for this effect are:

1. The bottleneck (producing the latent representation) in the architecture
   forces the model to generalize the input data.
2. We train using the mean squared error as the loss function,
   this has a smoothing effect as well as the learning goal for the model is
   effectively to produce low differences on average.
3. We use convolutions which slide across the data and hence can incur a
   smoothing effect.
"""

# %% [markdown]
"""
## **Exercise 04.1** Vary autoencoder hyper-parameters

The model predictions are actually not very good -- too much smoothing in some
parts of a signal, following the input signal too much in other parts. The same
could probably be acheived by a much simpler method such as a moving average :)

Try to improve this by varying the following parameters and observe their
effect on the reconstruction. Re-execute the cells above which define the model,
set the hyper-parameters and run the training.

Model architecture:

* nchannels
* nlayers (bigger means smaller latent space size)

Training:

* learning_rate
* max_epochs
"""

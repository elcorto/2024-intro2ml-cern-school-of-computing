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
from collections import defaultdict

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
layers to build the autoencoder. In particular, we follow this design for 2D
convolutions of images, adapted to our 1D case.

![image](img/guo_2017_cae.png)

Guo et al. "Deep Clustering with Convolutional Autoencoders", 2017 (https://doi.org/10.1007/978-3-319-70096-0_39)

Other resources:

* https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
* https://github.com/fquaren/Deep-Clustering-with-Convolutional-Autoencoders/blob/master/src/nets.py

"""


# %%
class MyEncoder(torch.nn.Module):
    def __init__(self, channels=[8, 16, 32], input_ndim=40, latent_ndim=10):
        super().__init__()
        self.layers = torch.nn.Sequential()

        channels = [1] + channels
        for old_n_channels, new_n_channels in zip(channels[:-1], channels[1:]):
            self.layers.append(
                torch.nn.Conv1d(
                    in_channels=old_n_channels,
                    out_channels=new_n_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                    stride=2,
                )
            )
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Flatten())

        # Calculate in_features for Linear layer
        dummy_X = torch.empty(1, 1, input_ndim, device="meta")
        dummy_out = self.layers(dummy_X)
        in_features = dummy_out.shape[-1]

        # Compress conv results into latent space
        self.layers.append(
            torch.nn.Linear(
                in_features=in_features,
                out_features=latent_ndim,
            )
        )

    def forward(self, x):
        # Convolutions in torch require an explicit channel dimension to be
        # present in the data, in other words:
        #   inputs of size (batch_size, 40) do not work,
        #   inputs of size (batch_size, 1, 40) do work
        if len(x.shape) == 2:
            return self.layers(torch.unsqueeze(x, dim=1))
        else:
            return self.layers(x)


# %% [markdown]
"""
This encoder is not yet trained, so its weights are random. Still, lets apply
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
print(model_summary(enc, input_size=X[:1, ...].shape))

# %% [markdown]
"""
The encoder takes a tensor of shape `[batch_size, 40]` (or `[batch_size, 1,
40]` with a channel dimension) and compresses that to a latent `h` of shape
`[batch_size, latent_ndim]`. Above, we used `batch_size=1` when calling
`model_summary()`.

The encoder has been constructed. Now, we need to add a decoder object to
reconstruct from the latent space.
"""


# %%
class MyDecoder(torch.nn.Module):
    def __init__(self, channels=[32, 16, 8], latent_ndim=10, output_ndim=40):
        super().__init__()
        self.layers = torch.nn.Sequential()

        # Architecture of the full autoencoder. Used here to help explain the
        # calculation of smallest_conv_ndim and linear_ndim.
        #
        # With channels=[32, 16, 8], latent_ndim=10, output_ndim=40, we have
        #
        #   smallest_conv_ndim = 5 = 40 // 2**3
        #
        # since we reduce input_ndim = output_ndim = 40 by factor of 2 in each
        # of the len(channels) = 3 conv steps in the encoder because of
        # Conv1d(..., stride=2).
        #
        # Further, we have
        #
        #   linear_ndim = 160 = 32 * 5
        #
        # Layer (type:depth-idx)                   Input Shape               Output Shape
        # ================================================================================
        # MyAutoencoder                            [1, 40]                   [1, 40]
        # ├─MyEncoder: 1-1                         [1, 40]                   [1, 10]
        # │    └─Sequential: 2-1                   [1, 1, 40]                [1, 10]
        # │    │    └─Conv1d: 3-1                  [1, 1, 40]                [1, 8, 20]
        # │    │    └─ReLU: 3-2                    [1, 8, 20]                [1, 8, 20]
        # │    │    └─Conv1d: 3-3                  [1, 8, 20]                [1, 16, 10]
        # │    │    └─ReLU: 3-4                    [1, 16, 10]               [1, 16, 10]
        # │    │    └─Conv1d: 3-5                  [1, 16, 10]               [1, 32, 5]
        # │    │    └─ReLU: 3-6                    [1, 32, 5]                [1, 32, 5]
        # │    │    └─Flatten: 3-7                 [1, 32, 5]                [1, 160]
        # │    │    └─Linear: 3-8                  [1, 160]                  [1, 10]
        # ├─MyDecoder: 1-2                         [1, 10]                   [1, 40]
        # │    └─Sequential: 2-2                   [1, 10]                   [1, 40]
        # │    │    └─Linear: 3-9                  [1, 10]                   [1, 160]
        # │    │    └─ReLU: 3-10                   [1, 160]                  [1, 160]
        # │    │    └─Unflatten: 3-11              [1, 160]                  [1, 32, 5]
        # │    │    └─ConvTranspose1d: 3-12        [1, 32, 5]                [1, 16, 10]
        # │    │    └─ReLU: 3-13                   [1, 16, 10]               [1, 16, 10]
        # │    │    └─ConvTranspose1d: 3-14        [1, 16, 10]               [1, 8, 20]
        # │    │    └─ReLU: 3-15                   [1, 8, 20]                [1, 8, 20]
        # │    │    └─ConvTranspose1d: 3-16        [1, 8, 20]                [1, 1, 40]
        # │    │    └─Flatten: 3-17                [1, 1, 40]                [1, 40]
        # ================================================================================

        smallest_conv_ndim = output_ndim // (2 ** len(channels))
        linear_ndim = channels[0] * smallest_conv_ndim

        # Decompress latent
        self.layers.append(
            torch.nn.Linear(
                in_features=latent_ndim,
                out_features=linear_ndim,
            )
        )
        self.layers.append(torch.nn.ReLU())

        # Reshape for conv upsampling
        self.layers.append(
            torch.nn.Unflatten(1, (channels[0], smallest_conv_ndim))
        )

        channels = channels + [1]
        for old_n_channels, new_n_channels in zip(channels[:-1], channels[1:]):
            self.layers.append(
                torch.nn.ConvTranspose1d(
                    in_channels=old_n_channels,
                    out_channels=new_n_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="zeros",
                    stride=2,
                    output_padding=1,
                )
            )
            self.layers.append(torch.nn.ReLU())

        # Remove last ReLU
        self.layers.pop(-1)

        # Remove channel dim (default is Flatten(..., start_dim=1))
        self.layers.append(torch.nn.Flatten())

    def forward(self, x):
        return self.layers(x)


# %%
dec = MyDecoder()

X_prime = dec(latent_h)
assert (
    X_prime.squeeze(1).shape == X_test.shape
), f"{X_prime.squeeze(1).shape} != {X_test.shape}"

print(f"{latent_h[:1, ...].shape=}")
print(model_summary(dec, input_size=latent_h[:1, ...].shape))

# %% [markdown]
"""
Now we have now all the lego bricks in place to compose an autoencoder. We do
this by combining the encoder and decoder in yet another `torch.nn.Module`.
"""


# %%
class MyAutoencoder(torch.nn.Module):
    def __init__(
        self, enc_channels=[8, 16, 32], latent_ndim=10, input_ndim=40
    ):
        super().__init__()

        self.enc = MyEncoder(
            channels=enc_channels,
            input_ndim=input_ndim,
            latent_ndim=latent_ndim,
        )
        self.dec = MyDecoder(
            channels=enc_channels[::-1],
            latent_ndim=latent_ndim,
            output_ndim=input_ndim,
        )

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
print(model_summary(model, input_size=X[:1, ...].shape))


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
and clean data points using `StackDataset`. We will look at the first `nrows *
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
    optimizer,
    loss_func,
    train_dataloader,
    test_dataloader,
    max_epochs,
    log_every=5,
    use_gpu=False,
    logs=defaultdict(list),
):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.train()

    for epoch in range(max_epochs):
        # For calculating loss averages in one epoch
        train_loss_epoch_sum = 0.0
        test_loss_epoch_sum = 0.0

        for train_noisy, train_clean in train_dataloader:
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
            train_loss = loss_func(
                X_prime_train, X_train_clean.squeeze().to(device)
            )

            # compute gradient
            train_loss.backward()

            # apply weight update rule
            optimizer.step()

            # set gradients to 0
            optimizer.zero_grad()

            train_loss_epoch_sum += train_loss.item()

        for test_noisy, test_clean in test_dataloader:
            # Discard labels if using StackDataset
            if isinstance(test_noisy, Sequence):
                X_test_noisy = test_noisy[0]
                X_test_clean = test_clean[0]
            else:
                X_test_noisy = test_noisy
                X_test_clean = test_clean

            X_prime_test = model(X_test_noisy.to(device))
            test_loss = loss_func(
                X_prime_test, X_test_clean.squeeze().to(device)
            )
            test_loss_epoch_sum += test_loss.item()

        logs["train_losses"].append(
            train_loss_epoch_sum / len(train_dataloader)
        )
        logs["test_losses"].append(test_loss_epoch_sum / len(test_dataloader))

        if (epoch + 1) % log_every == 0 or (epoch + 1) == max_epochs:
            print(
                f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.4f}; test loss {test_loss.mean():03.4f}"
            )
    return logs


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

model = MyAutoencoder(enc_channels=[8, 16, 32])
print(
    model_summary(model, input_size=next(iter(train_dataloader))[0][0].shape)
)

learning_rate = 1e-3
max_epochs = 50
log_every = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()

# Initialize empty loss logs once.
logs = defaultdict(list)

# %% [markdown]
"""
Run training.

Note that if you re-execute this cell with*out* reinstantiating `model` above,
you will continue training with the so-far best model as start point. Also, we
append loss histories to `logs`.
"""

# %%
logs = train_autoencoder(
    model=model,
    optimizer=optimizer,
    loss_func=loss_func,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=max_epochs,
    log_every=log_every,
    logs=logs,
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
ax.plot(logs["train_losses"], color="b", label="train")
ax.plot(logs["test_losses"], color="orange", label="test")
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
Also, looking at the loss plot, it seems that the training is not yet
converged.

Try to improve this by varying the following parameters and observe their
effect on the reconstruction. Re-execute the cells above which define the model,
set the hyper-parameters and run the training.

Training:

* `max_epochs`: try training for 200 epochs
* `learning_rate`: try 10x and 1/10 of the value above

Model architecture:

* `enc_channels`: try more channels per convolution and/or deeper models, such
   as `[32,64,128]` (more) or `[16,32,64,128,256]` (deeper)
* `latent_ndim`: the default is 10, try setting it to 2 or 20, what happens?
"""

# %% [markdown]
"""
## Visualize the latent space
"""

# %%
with torch.no_grad():
    grid_data = gridspec.GridSpec(nrows=5, ncols=2)
    grid_latent = gridspec.GridSpec(nrows=5, ncols=2)

    fig_data = plt.figure(
        figsize=(5 * grid_data.ncols, 5 * grid_data.nrows),
        layout="tight",
    )
    fig_latent = plt.figure(
        figsize=(5 * grid_latent.ncols, 5 * grid_latent.nrows),
        layout="tight",
    )

    axs_data = []
    for label, gs in enumerate(grid_data):
        axs_data.append(fig_data.add_subplot(gs))
        axs_data[-1].set_title(f"clean data, {label=}")
    axs_latent = []
    for label, gs in enumerate(grid_latent):
        axs_latent.append(fig_latent.add_subplot(gs))
        axs_latent[-1].set_title(f"latent h, {label=}")

    # https://matplotlib.org/stable/gallery/color/color_cycle_default.html
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    assert len(colors) == 10

    for test_noisy, test_clean in test_dataloader:
        X_test_noisy, y_test_noisy = test_noisy
        X_test_clean, y_test_clean = test_clean
        assert (y_test_noisy == y_test_clean).all()
        for idx in range(len(y_test_clean)):
            y = y_test_clean[idx]
            axs_data[y].plot(X_test_clean[idx].squeeze(), color=colors[y])
            axs_latent[y].plot(
                model.enc(X_test_noisy[idx]).squeeze(),
                color=colors[y],
            )

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

First, let's prepare a dataset, which is contains a signal we are interested in
and the noise.
"""

# %%
import numpy as np
import torch

from utils import model_summary

np.random.seed(13)
torch.random.manual_seed(12)

# %%
from mnist1d.data import get_dataset_args, make_dataset

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

# %% [markdown]
"""
Now, let's plot the data which we would like to use.
"""

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)

for sample in range(10):
    col = sample % 5
    row = sample // 5
    ax[row, col].plot(X[sample, ...], label="noisy")
    ax[row, col].plot(cleanX[sample, ...], label="clean", color="green")
    label = y[sample]
    ax[row, col].set_title(f"label {label}")
    if row == 1:
        ax[row, col].set_xlabel(f"samples / a.u.")
    if col == 0:
        ax[row, col].set_ylabel(f"intensity / a.u.")
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
   **latent representation** of your input data
3. **the decoder** on the right: reconstructs the output from the latent code

The task of the autoencoder is to reconstruct the input as best as possible.
This task is far from easy, as the autoencoder is forced to shrink the data
into the latent space.
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
            x = torch.unsqueeze(x, dim=1)
        return self.layers(x)


# %%
enc = MyEncoder()

# convert input data to torch.Tensor
Xt = torch.from_numpy(X)

# convert to float
Xt = Xt.float()

# extract only first 8 samples for testing
Xtest = Xt[:8, ...]

latent_h = enc(Xtest)

assert (
    latent_h.shape[-1] < Xtest.shape[-1]
), f"{latent_h.shape[-1]} !< {Xtest.shape[-1]}"

print(f"{Xt[:1, ...].shape=}")
model_summary(enc, input_size=Xt[:1, ...].shape)

# %% [markdown]
"""
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

    def forward(self, x):
        # convolutions in torch require an explicit channel dimension to be
        # present in the data in other words:
        # inputs of size (nbatch, 40) do not work,
        # inputs of size (nbatch, 1, 40) do work
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)

        return self.layers(x)


# %%
dec = MyDecoder()

Xt_prime = dec(latent_h)
assert (
    Xt_prime.squeeze(1).shape == Xtest.shape
), f"{Xt_prime.squeeze(1).shape} != {Xtest.shape}"

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
We can test our autoencoder works as expected similar to what we did above.
"""

# %%
model = MyAutoencoder()
Xt_prime = model(Xtest)

assert (
    Xt_prime.squeeze(1).shape == Xtest.shape
), f"{Xt_prime.squeeze(1).shape} != {Xtest.shape}"
print(f"autoencoder is ready to train!")

print(f"{Xt[:1, ...].shape=}")
model_summary(model, input_size=Xt[:1, ...].shape)


# %% [markdown]
"""
## **Exercise 04.1** MLPs for an autoencoder

We have so far built up our autoencoder with convolutional operations only. The
same can be done with `torch.nn.Linear` layers only. **Please code an encoder
and decoder that only require the use of `torch.nn.Linear` layers!** Keep the
signature of the `self.__init__` methods the same.
"""


# %% jupyter={"source_hidden": true}
# 04.1 Solution
class MyLinearEncoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16, inputdim=40):
        super().__init__()
        self.layers = torch.nn.Sequential()
        indim = inputdim

        # shrink input width by 2x
        outdim = inputdim // 2

        for i in range(nlayers - 1):
            self.layers.append(torch.nn.Linear(indim, outdim))

            # shrink input width by 2x
            indim = outdim
            outdim = indim // 2

            if i != (nlayers - 2):
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class MyLinearDecoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16, inputdim=10):
        super().__init__()
        self.layers = torch.nn.Sequential()
        indim = inputdim

        # expand input width by 2x
        outdim = inputdim * 2

        for i in range(nlayers - 1):
            self.layers.append(torch.nn.Linear(indim, outdim))

            indim = outdim
            # expand input width by 2x
            outdim = indim * 2

            # no relu for last layer
            if i != (nlayers - 2):
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        return self.layers(x)


lenc = MyLinearEncoder()
ldec = MyLinearDecoder()

# watch out, as we don't use convolutions, we don't need the extra dimension
# to denominate the channel number
Xtest_ = Xtest
latent_h_ = lenc(Xtest_)
assert latent_h_.shape == (
    8,
    10,
), f"{latent_h_.shape} is not (8,10) as expected"
Xtest_prime_ = ldec(latent_h_)
assert (
    Xtest_prime_.shape == Xtest_.shape
), f"{Xtest_prime_.shape} != {Xtest_.shape}"


# %% [markdown]
"""
## Training an autoencoder

Training the autoencoder works in the same line as training for regression from the last episode.

1. create the dataset
2. create the loaders
3. setup the model
4. setup the optimizer
5. loop through epochs
"""

# %%
from torch.utils.data import DataLoader
from utils import MNIST1D

# noisy data
training_noisy = MNIST1D(mnist1d_args=noisy_config, train=True)
test_noisy = MNIST1D(mnist1d_args=noisy_config, train=False)

# clean data
training_clean = MNIST1D(mnist1d_args=clean_config, train=True)
test_clean = MNIST1D(mnist1d_args=clean_config, train=False)

# stacked as paired sequences
training_data = torch.utils.data.StackDataset(training_noisy, training_clean)
test_data = torch.utils.data.StackDataset(test_noisy, test_clean)

train_dataloaders = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloaders = DataLoader(test_data, batch_size=64, shuffle=True)

nsamples = len(training_noisy) + len(test_noisy)
assert (
    nsamples == 4_000
), f"number of samples for MNIST1D is not 4_000 but {nsamples}"

model = MyAutoencoder(nchannels=32)
print(f"training conv autoencoder")

learning_rate = 1e-3
max_epochs = 20
log_every = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # our loss function


# %%
# write the training loop
def train_autoencoder(
    model,
    opt,
    crit,
    train_dataloader,
    test_dataloader,
    max_epochs,
    log_every=5,
):
    results = {"train_losses": [], "test_losses": []}
    ntrainsteps = len(train_dataloader)
    nteststeps = len(test_dataloader)
    train_loss, test_loss = (
        torch.zeros((ntrainsteps,)),
        torch.zeros((nteststeps,)),
    )

    for epoch in range(max_epochs):
        # perform training for one epoch
        for idx, (noisy, clean) in enumerate(train_dataloader):
            # we discard the labels by assigning them to _
            noisy_x, _ = noisy
            clean_x, _ = clean

            # forward pass
            X_prime = model(noisy_x)

            # compute loss
            loss = crit(X_prime, clean_x)

            # compute gradient
            loss.backward()

            # apply weight update rule
            opt.step()

            # set gradients to 0
            opt.zero_grad()

            train_loss[idx] = loss.item()

        for idx, (test_noisy, test_clean) in enumerate(test_dataloader):
            # we discard the labels by assigning them to _
            test_noisy_x, _ = noisy
            test_clean_x, _ = clean

            X_prime_test = model(test_noisy_x)
            loss_ = crit(X_prime_test, test_clean_x)
            test_loss[idx] = loss_.item()

        results["train_losses"].append(train_loss.mean())
        results["test_losses"].append(test_loss.mean())

        if epoch % log_every == 0 or (epoch + 1) == max_epochs:
            print(
                f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.4f}; test loss {test_loss.mean():03.4f}"
            )
    return results


results = train_autoencoder(
    model,
    optimizer,
    criterion,
    train_dataloaders,
    test_dataloaders,
    max_epochs,
    log_every,
)
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(results["train_losses"], color="b", label="train")
ax[0].plot(results["test_losses"], color="orange", label="test")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("average MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()

index = 0
# perform prediction again
last_x, last_y = test_noisy[index]
last_x_prime = model(last_x.unsqueeze(0))

# prepare tensors for plotting
last_in = last_x.detach().squeeze().numpy()
last_out = last_x_prime.detach().squeeze().numpy()

# obtain reference test data
clean_x, clean_y = test_clean[index]
clean_in = clean_x.detach().squeeze().numpy()

ax[1].plot(last_in, color="b", label="test input")
ax[1].plot(last_out, color="orange", label="test prediction")
ax[1].plot(clean_in, color="green", linestyle="--", label="clean")
ax[1].set_xlabel("samples / a.u.")
ax[1].set_ylabel("intensity / a.u.")
ax[1].set_title(f"Conv-based Autoencoder, label = {last_y.detach().item()}")
ax[1].legend()

fig.savefig("mnist1d_noisy_conv_autoencoder_training.svg")

# %% [markdown]
"""
We can see that the autoencoder smoothed the input signal when producing a
reconstruction. This denoising effect can be quite helpful in practice. The
core reasons for this effect are:

1. the bottleneck (producing the latent representation) in the architecture
   forces the model to generalize the input data
2. we train using the L2 norm (or mean squared error) as the loss function,
   this has a smoothing effect as well as the learning goal for the model is
   effectively to produce low differences on average
3. we use convolutions which slide across the data and hence can incur a smoothing effect

If you try the last cell with different values for `index` you will also see
that the autoencoder did not memorize the data or magically learned how to
reproduce the denoised `clean` data.
"""

# %% [markdown]
"""
## **Exercise 04.2** MLPs for an autoencoder for good

Rewrite the MyAutoencoder class to use the encoder/decoder classes which employ
`torch.nn.Linear` layers only. Rerun the training with them! Do you observe a
difference in the reconstruction?
"""


# %% jupyter={"source_hidden": true}
# 04.2 Solution
class MyLinearAutoencoder(torch.nn.Module):
    def __init__(self, nlayers=3, nchannels=16):
        super().__init__()

        self.enc = MyLinearEncoder(nlayers, nchannels)
        self.dec = MyLinearDecoder(nlayers, nchannels)

    def forward(self, x):
        # construct the latents
        h = self.enc(x)

        # perform reconstruction
        x_prime = self.dec(h)

        return x_prime


# setup model and optimizer
lmodel = MyLinearAutoencoder(nchannels=32)
print(f"training dense autoencoder")

loptimizer = torch.optim.AdamW(lmodel.parameters(), lr=learning_rate)

# run training
lresults = train_autoencoder(
    lmodel,
    loptimizer,
    criterion,
    train_dataloaders,
    test_dataloaders,
    max_epochs,
    log_every,
)

# viz the results
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(lresults["train_losses"], color="b", label="train")
ax[0].plot(lresults["test_losses"], color="orange", label="test")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("average MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()

# perform prediction again
last_x_prime = lmodel(last_x.unsqueeze(0))

# prepare tensors for plotting
last_out = last_x_prime.detach().squeeze().numpy()

ax[1].plot(last_in, color="b", label="test input")
ax[1].plot(last_out, color="orange", label="test prediction")
ax[1].plot(clean_in, color="green", linestyle="--", label="clean")
ax[1].set_xlabel("samples / a.u.")
ax[1].set_ylabel("intensity / a.u.")
ax[1].set_title(f"Linear Autoencoder, label = {last_y.detach().item()}")
ax[1].legend()

fig.savefig("mnist1d_noisy_linear_autoencoder_training.svg")

# %% [markdown] jupyter={"source_hidden": true}
"""
Congratulations, you have successfully trained an all-linear autoencoder! You
can see that the denoising effect is not as strong as with the convolutional
operations. One thing is certain however, also the linear layer based
autoencoder is capable of retaining the signal "peaks". Note, some
generalizations based on this are premature at this point.

To draw more conclusions, here are some things to try while retaining the
number of parameters of both autoencoders the same:

- train on more data
- use different activation functions
- add more layers
- optimize the hyperparameters for training (learning_rate, number of epochs, ...)
"""

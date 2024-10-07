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
# Representation Learning

Effective Machine Learning is often about finding a good and flexible model
that can represent high-dimensional data well. The autoencoder can be such an
architecture.

Here we first investigate the autoencoder's latent space. Then we train
a classification CNN, which has a completely different task. Instead of
learning to compress (and denoise) the data, it must classify the inputs by
label. We will look at its latent representations of the data. Does
it learn to pay attention to the same data characteristics to solve its task?

**This notebook requires the files `X_latent_h.npy` and `y_latent_h.py` written
by the autoencoder notebook. If those are not resent, please (re-)run this
first.**
"""


# %%
# First we need to repeat some code from the previous notebook. We could of
# course put everything into modules and import it, which would be the correct
# way to do it in production, but here we don't, for didactic purposes.

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from mnist1d.data import get_dataset_args, make_dataset

from utils import model_summary, MNIST1D, get_label_colors

np.random.seed(13)
torch.random.manual_seed(12)

# %%
# disable noise for a clear reference
clean_config = get_dataset_args()
clean_config.iid_noise_scale = 0
clean_config.corr_noise_scale = 0
clean_config.seed = 40
clean_mnist1d = make_dataset(clean_config)
X_clean, y_clean = clean_mnist1d["x"], clean_mnist1d["y"]

# use iid noise only for the time being
noisy_config = get_dataset_args()
noisy_config.iid_noise_scale = 0.05
noisy_config.corr_noise_scale = 0
noisy_config.seed = 40
noisy_mnist1d = make_dataset(noisy_config)
X_noisy, y_noisy = noisy_mnist1d["x"], noisy_mnist1d["y"]

# We use the same random seed for clean_config and noisy_config, so this must
# be the same.
assert (y_clean == y_noisy).all()

# Convert numpy -> torch for usage in next cells. For training, we will build a
# DataLoader later.
X_noisy = torch.from_numpy(X_noisy).float()
X_test = X_noisy[:8, ...]


# %%
# This function is much simpler now, since we train on clean inputs and
# targets.
#
def get_dataloaders(batch_size=64):
    dataset_train_clean = MNIST1D(mnist1d_args=clean_config, train=True)
    dataset_test_clean = MNIST1D(mnist1d_args=clean_config, train=False)
    assert len(dataset_train_clean) == 3600
    assert len(dataset_test_clean) == 400

    train_dataloader = DataLoader(
        dataset_train_clean, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset_test_clean, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader


# %%
# Load autoencoder latent h produced by the autoencoder notebook.
X_latent_h = np.load("X_latent_h.npy")
y_latent_h = np.load("y_latent_h.npy")


# %% [markdown]
"""
## Projection of the autoencoder latent `h` in 2D

In the autoencoder lesson, we plotted the latent `h` and found it hard to find
some structure by visual inspection.

Let's now project the latent representations `h` of dimension 10 into a 2D space
and see if we can find some structure there. For this we use [t-distributed
Stochastic Neighbor Embedding
(t-SNE)](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne),
[Uniform Manifold Approximation and Projection for Dimension Reduction
(UMAP)](https://umap-learn.readthedocs.io)
as well as Isomap as one additional method of the many that `scikit-learn`
offers.
"""

# %%
# We'll cache things in here that we'd like to reuse instead of recomputing
# them.
vis_cache = defaultdict(dict)
default_emb_name = "umap"

emb_methods = dict(
    tsne=TSNE(n_components=2, random_state=23),
    umap=UMAP(n_components=2, random_state=23),
    isomap=Isomap(n_components=2),
)

ncols = 1
nrows = len(emb_methods)
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
)
label_colors = get_label_colors(y_latent_h)
X_scaled = StandardScaler().fit_transform(X_latent_h)
for (emb_name, emb), ax in zip(emb_methods.items(), np.atleast_1d(axs)):
    print(f"processing: {emb_name}")
    X_emb2d = emb.fit_transform(X_scaled)
    ax.scatter(X_emb2d[:, 0], X_emb2d[:, 1], c=label_colors)
    ax.set_title(f"MNIST-1D latent h: {emb_name}")
    vis_cache["ae_latent_h"][emb_name] = dict(X_emb2d=X_emb2d, y=y_latent_h)


fig.savefig("mnist1d_ae_latent_embeddings_2d.svg")

# %% [markdown]
"""
If your autoencoder model is big enough and training is converged, you should
see now that overall, there is no clear clustering into groups **by label**
(the different colors) for all classes. Instead, we find some classes which are
represented by a number of smaller "sub-clusters" which share the same label
(esp. in the t-SNE and UMAP plots). Other classes don't show sub-clusters, but are
instead scattered all over the place.

In summary, there is definitely structure in the latent `h` representations
of the data, just not one that can be easily mapped to one class label per cluster.
So why is that? We will investigate this now in more detail.

Note: Dimensionality reduction is a tricky business which by construction is a
process where information is lost, while trying to retain the most prominent
parts. Also, each method has hyper-parameters that need to be explored before
over-interpreting any method's results. Still, if the model had learned to
produce very distinct embeddings `h` per class label, we would also expect to
see this even in a 2D space.

To gain more insights, we now compute additional 2D embeddings: We
project the MNIST-1D *inputs* of dimension 40 into a 2D space.
"""

# %%
cases = [
    dict(
        dset_name="MNIST-1D AE latent h, class labels",
        X=vis_cache["ae_latent_h"][default_emb_name]["X_emb2d"],
        y=vis_cache["ae_latent_h"][default_emb_name]["y"],
        compute=False,
    ),
    dict(
        dset_name="MNIST-1D input (clean), class labels",
        X=X_clean,
        y=y_clean,
        compute=True,
    ),
    dict(
        dset_name="MNIST-1D input (noisy), class labels",
        X=X_noisy,
        y=y_noisy,
        compute=True,
    ),
]

ncols = len(cases)
nrows = 1
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows)
)

for dct, ax in zip(cases, np.atleast_1d(axs)):
    dset_name = dct["dset_name"]
    X = dct["X"]
    y = dct["y"]
    compute = dct["compute"]
    print(f"processing: {dset_name}")
    if compute:
        X_emb2d = emb_methods[default_emb_name].fit_transform(
            StandardScaler().fit_transform(X)
        )
    else:
        X_emb2d = X
    ax.scatter(X_emb2d[:, 0], X_emb2d[:, 1], c=get_label_colors(y))
    n_unique_labels = len(np.unique(y))
    ax.set_title(f"{dset_name} \n#labels = {n_unique_labels}")

fig.savefig("mnist1d_embeddings_2d_compare.svg")


# %% [markdown]
"""
On the left we have the same 2D plot as before, a projection of the
latent `h` into 2D space. The middle and right plots show the 2D projections of the
40-dimensional inputs. We can make the following observations:

* The input embeddings (middle and right) look very similar, so the noise we
  added to the clean data is such that more than enough of the clean data
  characteristics are retained, which makes learning a denoising model
  possible in the first place.
* The embedding of the latent `h` and that of the inputs are similar in terms of which
  classes cluster (or not). Note that we project with t-SNE/UMAP/... 10
  dimensional and 40 dimensional data and hence the produced 2D *shapes* are
  not expected to be the same, as those have no meaning in those methods (see
  [this](https://scikit-learn.org/stable/modules/manifold.html#optimizing-t-sne))
  for more. Only the spatial distribution of the class colors is what matters.
* Recall that the inputs and the
  latent `h` look *very* different, yet their 2D representations are remarkably
  similar. This shows that the latent codes `h` indeed en**code** the
  characteristics of the data, even though their numbers (e.g. plotted in 1D)
  appear meaningless to us. Be reminded that, just as with a standard
  compression method (like xz, zip, lz4, ...) the compressed data looks nothing
  like the input. You need the compressed version *plus* the compression
  (encoder) and decompression (decoder) software. In our case, the autoencoder
  with its learned weights is the "software", applicable to this dataset.
* The left plot looks less fragmented (less sub-clusters) than even the
  embedding of the clean data (middle). This suggests that the latent `h` carry
  only essential information regarding the data characteristics, i.e. the
  autoencoder managed to remove data features that are not important.

But the question remains: Why don't we see one single cluster per class label?
Two hypotheses come to mind:

* The autoencoder was *not* trained to classify inputs by label, but to
  reconstruct and denoise them. Hence the model may learn to produce latent codes
  that help in doing that, and as a result may focus on other structural elements
  of the data than those which a classification model would use to discriminate
  between classes.

* Given the similarity of the input and latent `h`'s 2D embeddings, maybe the
  dataset itself is hard, in the sense that some classes can be separated by
  input data features (the ones that show up in sub-clusters), while other
  inputs with different class labels have in fact very similar data
  characteristics.
"""


# %% [markdown]
"""
## Classifying MNIST-1D

Similar to [MNIST](https://yann.lecun.com/exdb/mnist/), MNIST-1D can be used
for the task of classification where, given an input sequence, we
want to predict the class label `[0,1,...,9]` that the 1D sequence belongs to.

We now build a CNN classification model, the architecture of which is similar
to our encoder from before. The main difference is that after the convolutional
layers which do "feature learning" (learn what to pay attention to in the
input), we have a small MLP that solves the classification task. We will use
its hidden layer's activations as latent representations.
"""


# %%
class MyCNN(torch.nn.Module):
    def __init__(
        self,
        channels=[8, 16, 32],
        input_ndim=40,
        output_ndim=10,
        latent_ndim=64,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential()

        channels = [1] + channels
        for ii, (old_n_channels, new_n_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
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
            if ii < len(channels) - 2:
                self.layers.append(
                    torch.nn.Conv1d(
                        in_channels=new_n_channels,
                        out_channels=new_n_channels,
                        kernel_size=3,
                        padding=1,
                        padding_mode="replicate",
                        stride=1,
                    )
                )
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Flatten())

        dummy_X = torch.empty(1, 1, input_ndim, device="meta")
        dummy_out = self.layers(dummy_X)
        in_features = dummy_out.shape[-1]

        self.layers.append(
            torch.nn.Linear(
                in_features=in_features,
                out_features=2 * latent_ndim,
            )
        )

        self.layers.append(torch.nn.ReLU())

        # This layer will be used as the latent data representation of the CNN.
        self.layers.append(
            torch.nn.Linear(
                in_features=2 * latent_ndim,
                out_features=latent_ndim,
            )
        )

        self.final_layers = torch.nn.Sequential()

        self.final_layers.append(torch.nn.ReLU())

        self.final_layers.append(
            torch.nn.Linear(
                in_features=latent_ndim,
                out_features=output_ndim,
            )
        )

    def forward(self, x):
        # Convolutions in torch require an explicit channel dimension to be
        # present in the data, in other words:
        #   inputs of size (batch_size, 40) do not work,
        #   inputs of size (batch_size, 1, 40) do work
        if len(x.shape) == 2:
            latent_cnn = self.layers(torch.unsqueeze(x, dim=1))
        else:
            latent_cnn = self.layers(x)

        # In contrast to a standard forward() method, we return a tuple with
        # the normal output and the latent representation. We account for that
        # in the train function and in other places where we call model(X).
        return self.final_layers(latent_cnn), latent_cnn


# %% [markdown]
# Next we define a function that runs the training. This function is almost the
# same as the one used for the autoencoder. The differences are:
#
# * Instead of having noisy data `X_noisy` as input and `X_clean` as target
#   (autoencoder), we now have `X_clean` as input and `y_clean` (class labels) as
#   target (classification).
# * We also record the classification accuracy in the `logs` dict.

# %%
from sklearn.metrics import accuracy_score as accuracy


def train_classifier(
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

    for epoch in range(max_epochs):
        train_loss_epoch_sum = 0.0
        test_loss_epoch_sum = 0.0
        train_acc_epoch_sum = 0.0
        test_acc_epoch_sum = 0.0

        model.train()
        for X_train, y_train in train_dataloader:
            # forward pass, discard latent_cnn here
            y_pred_train_logits, _ = model(X_train.to(device))

            train_loss = loss_func(y_pred_train_logits, y_train.to(device))
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_epoch_sum += train_loss.item()
            train_acc_epoch_sum += accuracy(
                y_train, y_pred_train_logits.argmax(-1).cpu().numpy()
            )

        model.eval()
        for X_test, y_test in test_dataloader:
            y_pred_test_logits, _ = model(X_test.to(device))
            test_loss = loss_func(y_pred_test_logits, y_test.to(device))
            test_loss_epoch_sum += test_loss.item()
            test_acc_epoch_sum += accuracy(
                y_test, y_pred_test_logits.argmax(-1).cpu().numpy()
            )

        logs["train_loss"].append(train_loss_epoch_sum / len(train_dataloader))
        logs["test_loss"].append(test_loss_epoch_sum / len(test_dataloader))
        logs["train_acc"].append(train_acc_epoch_sum / len(train_dataloader))
        logs["test_acc"].append(test_acc_epoch_sum / len(test_dataloader))

        if (epoch + 1) % log_every == 0 or (epoch + 1) == max_epochs:
            print(
                f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.5f}; test loss {test_loss.mean():03.5f}"
            )
    return logs


# %%
batch_size = 64
train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)


# %%
# hyper-parameters that influence model and training
learning_rate = 5e-4
latent_ndim = 64

max_epochs = 50
channels = [32, 64, 128]

# Regularization parameter to prevent overfitting.
weight_decay = 0.1

# Defined above already. We skip this here since this is a bit slow. If you
# want to change batch_size (yet another hyper-parameter!) do it here or in the
# cell above where we called get_dataloaders().
##batch_size = 64
##train_dataloader, test_dataloader = get_dataloaders(
##    batch_size=batch_size
##)

model = MyCNN(channels=channels, latent_ndim=latent_ndim)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
loss_func = torch.nn.CrossEntropyLoss()

# Initialize empty loss logs once.
logs = defaultdict(list)

print(
    model_summary(model, input_size=next(iter(train_dataloader))[0][0].shape)
)

# %% [markdown]
"""
Run training.

Note that if you re-execute this cell with*out* reinstantiating `model` above,
you will continue training with the so-far best model as start point. Also, we
append loss histories to `logs`.
"""

# %%
logs = train_classifier(
    model=model,
    optimizer=optimizer,
    loss_func=loss_func,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=max_epochs,
    log_every=5,
    logs=logs,
)


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(logs["train_loss"], color="b", label="train")
ax[0].plot(logs["test_loss"], color="orange", label="test")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("average MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()

ax[1].plot(logs["train_acc"], color="b", label="train")
ax[1].plot(logs["test_acc"], color="orange", label="test")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("average Accuracy / a.u.")
ax[1].set_title("Accuracy")
ax[1].legend()

fig.savefig("mnist1d_cnn_loss_acc.svg")


# %% [markdown]
"""
Let's create a 2D projection of the CNN's latent representation.
"""

# %%
with torch.no_grad():
    X_latent_cnn = model(torch.from_numpy(X_clean).float())[1]
y_latent_cnn = y_clean

cases = [
    dict(
        dset_name="MNIST-1D AE latent h, class labels",
        X=vis_cache["ae_latent_h"][default_emb_name]["X_emb2d"],
        y=vis_cache["ae_latent_h"][default_emb_name]["y"],
        compute=False,
    ),
    dict(
        dset_name="MNIST-1D CNN latent, class labels",
        X=X_latent_cnn,
        y=y_latent_cnn,
        compute=True,
    ),
]

ncols = len(cases)
nrows = 1
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows)
)

for dct, ax in zip(cases, np.atleast_1d(axs)):
    dset_name = dct["dset_name"]
    X = dct["X"]
    y = dct["y"]
    compute = dct["compute"]
    print(f"processing: {dset_name}")
    if compute:
        X_emb2d = emb_methods[default_emb_name].fit_transform(
            StandardScaler().fit_transform(X)
        )
    else:
        X_emb2d = X
    ax.scatter(X_emb2d[:, 0], X_emb2d[:, 1], c=get_label_colors(y))
    n_unique_labels = len(np.unique(y))
    ax.set_title(f"{dset_name} \n#labels = {n_unique_labels}")

fig.savefig("mnist1d_cnn_latent_embeddings_2d.svg")

# %% [markdown]
"""
OK, well that's interesting! Now, which of the two hypotheses from above do you
think is correct? Let's discuss!
"""

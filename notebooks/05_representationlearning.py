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
architecture depending on its design and the input data. In practice, the
community has started to use the latent representation for all kinds of
applications.
"""


# %%
# First we need to repeat some code from the previous notebook. We could of
# course put everything into modules and import it, which would be the correct
# way to do it in production, but here we don't, for didactic purposes.

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import gridspec, pyplot as plt

from mnist1d.data import get_dataset_args, make_dataset

from utils import model_summary, MNIST1D, colors_10, get_label_colors

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


X_latent_h = np.load("X_latent_h.npy")
y_latent_h = np.load("y_latent_h.npy")


# %% [markdown]
"""
Let's project the latent representations `h` of dimension 10 into a 2D space
and see if we can find some structure there. For this we use [t-distributed
Stochastic Neighbor Embedding
(t-SNE)](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)
as well as Isomap as one additional method of the many that `scikit-learn`
offers.
"""

# %%
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

emb_methods = dict(
    tsne=TSNE(n_components=2, random_state=23),
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

fig.savefig("mnist1d_ae_latent_embeddings_2d.svg")

# %% [markdown]
"""
If your model is big enough and training is converged, you should see now that
overall, there is no clear clustering into groups **by label** (the different
colors) for all classes. Instead, we find many smaller
"sub-clusters" which share the same label (esp. in the t-SNE plot). In other
words, there is definitely structure in the learned latent `h` representations
of the data (else we'd see only uniform 2D
blobs), just not one that can be easily mapped to class labels. So why is that?

This may be because the autoencoder was *not* trained to classify inputs by
label, but to reconstruct and denoise them. Hence the model learns to produce
latent codes that help in doing that, and as a result may focus on other
structural elements of the data than those which a classification model would
use to discriminate between classes. We will investigate this now in more
detail.

Note: Dimensionality reduction is a tricky business which by construction is a
process where information is lost, while trying to retain the most prominent
parts. Also, each method has hyper-parameters that need to be explored before
over-interpreting any method's results. Still, if the model had learned to
produce very distinct embeddings, we would also expect to see this even in a 2D
space.

To gain more insights, we now compute additional t-SNE embeddings: We
project the MNIST-1D *inputs* of dimension 40 into a 2D space.
"""

# %%
from sklearn.cluster import HDBSCAN, KMeans


def cluster(X):
    print("Running clustering ...")
    cl = HDBSCAN(min_cluster_size=5, min_samples=1)
    ##cl = KMeans(n_clusters=10)
    cl.fit(StandardScaler().fit_transform(X))
    return cl.labels_


print("Running 2D embedding ...")
X_latent_h_emb2d = TSNE(n_components=2, random_state=23).fit_transform(
    StandardScaler().fit_transform(X_latent_h)
)

cases = [
    dict(
        dset_name="MNIST-1D latent h, class labels",
        X=X_latent_h_emb2d,
        y=y_latent_h,
        clustered=False,
        compute_emb2d=False,
    ),
    dict(
        dset_name="MNIST-1D input (clean), class labels",
        X=X_clean,
        y=y_clean,
        clustered=False,
        compute_emb2d=True,
    ),
    dict(
        dset_name="MNIST-1D input (noisy), class labels",
        X=X_noisy,
        y=y_noisy,
        clustered=False,
        compute_emb2d=True,
    ),
    ##dict(
    ##    dset_name="MNIST-1D latent h, cluster labels",
    ##    X=X_latent_h_emb2d,
    ##    y=cluster(X_latent_h),
    ##    clustered=True,
    ##    compute_emb2d=False,
    ##),
    ##dict(
    ##    dset_name="MNIST-1D input, cluster labels",
    ##    X=X_clean,
    ##    y=cluster(X_clean),
    ##    clustered=True,
    ##    compute_emb2d=True,
    ##),
]

ncols = len(cases)
nrows = 1
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
)

for dct, ax in zip(cases, np.atleast_1d(axs)):
    dset_name = dct["dset_name"]
    y = dct["y"]
    clustered = dct["clustered"]
    X = dct["X"]
    compute_emb2d = dct["compute_emb2d"]
    if compute_emb2d:
        X_emb2d = TSNE(n_components=2, random_state=23).fit_transform(
            StandardScaler().fit_transform(X)
        )
    else:
        X_emb2d = X
    print(f"processing: {dset_name}")
    if clustered:
        msk_clusters = y >= 0
        msk_no_clusters = y < 0
        y_clusters = y[msk_clusters]
        ax.scatter(
            X_emb2d[msk_clusters, 0],
            X_emb2d[msk_clusters, 1],
            c=get_label_colors(y_clusters),
        )
        ax.scatter(
            X_emb2d[msk_no_clusters, 0],
            X_emb2d[msk_no_clusters, 1],
            color="k",
            marker="+",
            alpha=0.2,
        )
        n_unique_labels = len(np.unique(y_clusters))
    else:
        ax.scatter(X_emb2d[:, 0], X_emb2d[:, 1], c=get_label_colors(y))
        n_unique_labels = len(np.unique(y))
    ax.set_title(f"{dset_name} \n#labels = {n_unique_labels}")

fig.savefig("mnist1d_embeddings_2d_compare.svg")


# %% [markdown]
"""
On the left we have the same 2D plot as before, a projection of the learned
latent `h` into 2D space. The middle and right plots show the t-SNE embedding of the
40-dimensional inputs. We can make the following observations:

* The input embeddings (middle and right) look very similar, so the noise we
  added to the clean data is such that more than enough of the clean data
  characteristics are retained, such that learning a denoising model is
  actually possible.
* The embedding of the latens `h` and that of the inputs are similar in terms of which
  classes cluster more (or not). Note that we embed with t-SNE 10 dimensional
  and 40 dimensional data and hence the produced 2D *shapes* are not *the same*
  as those have no meaning as such. Only the spatial distiribution of the class
  colors is what matters.
* Recall that the inputs and the
  latent `h` look *very* different, yet their 2D representations are remarkably
  similar. This shows that the latent codes `h` indeed en**code** the
  characteristics of the data, even though their numbers (e.g. plotted in 1D)
  appear meaningless to us. Be reminded that, just as with a standard
  compression method (like xz, zip, lz4, ...) the compressed data looks nothing
  like the input. You need the compressed version *plus* the compression
  (encoder) and decompression (decoder) software. In our case, the autoencoder
  with its learned weights is the "software", applicable to this dataset.
* All plots show again sub-clusters that share labels, rather than one cluster
  per class. This shows that MNIST-1D is in fact a pretty hard nut to crack,
  when used as a classification dataset (in contrast to MNIST, say, which can
  be solved even by simple linear models). As we stated earlier, our
  autoencoder learns the overall characteristics of the data to solve the
  *reconstruction* task, independent of the labels, which are not used in
  training.
* The left plot looks less fragmented (less sub-clusters) than even the
  embedding of the clean data (middle). This suggests that the latent `h` carry
  only essential information regarding the data characteristics, i.e. the
  autoencoder managed to remove data features that are not important.
"""


# %% [markdown]
"""
## Classifying MNIST1D

Similar to [MNIST](https://yann.lecun.com/exdb/mnist/), `mnist1d` can be used
for the task of classification. In other words, given an input sequence, we
only want to predict the class label `[0,1,...,9]` that the image belongs to.
Classification has been one of the driving forces behind progress in machine
learning since [ImageNet 2012]() - for better or worse. In science,
classification is used rarely.
"""


class MyCNN(torch.nn.Module):
    def __init__(self, channels=[25, 25, 25], input_ndim=40, output_ndim=10):
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
                out_features=128,
            )
        )

        self.last_linear = torch.nn.Linear(
            in_features=128,
            out_features=output_ndim,
        )

    def forward(self, x):
        # Convolutions in torch require an explicit channel dimension to be
        # present in the data, in other words:
        #   inputs of size (batch_size, 40) do not work,
        #   inputs of size (batch_size, 1, 40) do work
        if len(x.shape) == 2:
            hidden_out = self.layers(torch.unsqueeze(x, dim=1))
        else:
            hidden_out = self.layers(x)
        return self.last_linear(hidden_out), hidden_out


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
            # forward pass, discard hidden_out here
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
latent_ndim = 10

max_epochs = 50
channels = [25, 25, 25]

# Regularization parameter to prevent overfitting. This is the AdamW
# optimizer's default value.
weight_decay = 0.01

# Defined above already. We skip this here since this is a bit slow. If you
# want to change batch_size (yet another hyper-parameter!) do it here or in the
# cell above where we called get_dataloaders().
##batch_size = 64
##train_dataloader, test_dataloader = get_dataloaders(
##    batch_size=batch_size
##)

model = MyCNN(channels=channels)
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
ax[0].set_ylabel("avergage MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()

ax[1].plot(logs["train_acc"], color="b", label="train")
ax[1].plot(logs["test_acc"], color="orange", label="test")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("avergage Accuracy / a.u.")
ax[1].set_title("Accuracy")
ax[1].legend()

fig.savefig("mnist1d_cnn_loss_acc.svg")


# %%
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

with torch.no_grad():
    X_latent_cnn = model(torch.from_numpy(X_clean).float())[1]
y_latent_cnn = y_clean

emb_methods = dict(
    tsne=TSNE(n_components=2, random_state=23),
    isomap=Isomap(n_components=2),
)

ncols = 1
nrows = len(emb_methods)
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
)
label_colors = get_label_colors(y_latent_cnn)
X_scaled = StandardScaler().fit_transform(X_latent_cnn)
for (emb_name, emb), ax in zip(emb_methods.items(), np.atleast_1d(axs)):
    print(f"processing: {emb_name}")
    X_emb2d = emb.fit_transform(X_scaled)
    ax.scatter(X_emb2d[:, 0], X_emb2d[:, 1], c=label_colors)
    ax.set_title(f"MNIST-1D CNN latent: {emb_name}")

fig.savefig("mnist1d_cnn_latent_embeddings_2d.svg")

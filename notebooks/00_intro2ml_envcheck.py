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
# # Environment Check
#
# This notebook is intended for all learners to check their environment. Click
# through it to see if your environment is setup correctly.

# %%
import torch

print("torch installed, ", torch.__version__)

# %%
import seaborn

print("seaborn installed, ", seaborn.__version__)

# %% [markdown]
#
# For most of the notebooks, we require the handy
# [MNIST1D](https://github.com/greydanus/mnist1d) dataset which is both small,
# versatile and tricky. The dataset can be installed as a `pip` package.
#
# To do so, create a new cell below and run the following command in it:
#
# ```
# !python -m pip install --user mnist1d
# ```

# %%
import mnist1d

if hasattr(mnist1d, "get_dataset"):
    print("mnist1d installed!")
else:
    print("mnist1d NOT installed. See instructions for details.")

# %% [markdown]
#
# If you were able to run the notebook until this point without any errors, you are

# %%
print("READY to GO!")

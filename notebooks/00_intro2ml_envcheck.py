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
# Environment Check

This notebook is intended for all learners to check their environment.

This should print something like:

```
jupytext            : found, version 1.16.4
ipykernel           : found, version 6.29.5
notebook            : found, version 6.4.13
torch               : found, version 2.4.1+rocm6.1
seaborn             : found, version 0.13.2
mnist1d             : found, (no version info)
sklearn             : found, version 1.4.2
torchinfo           : found, version 1.8.0
```
"""

# %%
import os

from utils import import_check

here = os.path.abspath(os.path.expanduser('.'))

found = False
for path in [here, f"{here}/.."]:
    fn = f"{path}/requirements.txt"
    if os.path.exists(fn):
        print(f"using: {fn}")
        import_check(fn)
        found = True
        break

if not found:
    raise Exception("requirements.txt not found")

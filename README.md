# Teaching material for the course "Introduction to machine learning" at the 2024 THRILL summer school

https://indico.gsi.de/event/19869/

## Installation instructions for local execution

> [!IMPORTANT]
> Please install the software components as described below **before** the
> course.


* If you don't have a local Python installation, try
  [miniconda](https://docs.anaconda.com/miniconda) or install Python with
  [`uv`](https://docs.astral.sh/uv)
  by
  * first [installing `uv`
    itself](https://docs.astral.sh/uv/getting-started/installation/)
  * [use `uv python install`](https://docs.astral.sh/uv/guides/install-python/)

  Both `miniconda` and `uv` work on MacOS, Linux and Windows.

* create a `venv`:
  * if you installed Python via `uv`

    ```sh
    uv venv thrill24
    source thrill24/bin/activate
    ```

  * else

    ```sh
    python -m venv thrill24
    source thrill24/bin/activate
    ```

> [!NOTE]
> On Windows, the `activate` script is usually `thrill24/Scripts/activate` and
> has to be executed, e.g. in PowerShell as `.\thrill24\Scripts\activate`.

* clone this repo: `git clone https://github.com/elcorto/2024-thrill-school-machine-learning`
* change into the repo root folder: `cd 2024-thrill-school-machine-learning`
* install CPU-only [`torch`](https://pytorch.org/) (optional) and other dependencies (required)

  ```sh
  uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
  uv pip install -r ./requirements.txt
  ```

  or

  ```sh
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
  python -m pip install -r ./requirements.txt
  ```

* If you do **not** have a tool to run Jupyter notebooks like [JupyterLab,
  Jupyter Notebook](https://jupyter.org/) or [VS
  Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
  already installed, then
  we recommend to install Jupyter following [these
  instructions](https://jupyter.org/install).

### Alternatives to a local install

* Go to https://mybinder.org/ and point it to this repo's URL. It will create a
  docker image, install all dependencies from `requirements.txt` and launch a
  JupyterLab (might take a while).
* [Google Colab](https://colab.research.google.com) it **not** really an
  option:
  * it requires a Google account
  * it doesn't seem to have native support for installing dependencies from a
    `requirements.txt` file and using those for all notebooks, you may need to
    [install them from within each
    notebook](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)
  * it seems to focus on stand-alone notebooks while ours depend on local Python
    modules containing utility code

## Check your software stack

This repo has a small utility prepared which can check if your software
environment is ready. Either run

```sh
python notebooks/00_intro2ml_envcheck.py
```

or open the paired notebook `notebooks/00_intro2ml_envcheck.ipynb` with
Jupyter, read the instructions and execute all cells.

# Teaching this material

If you are an instructor, please see the [instructor notes](FOR_INSTRUCTORS.md).

# References

This material is based on
https://github.com/psteinb/2024-intro2ml-cern-school-of-computing. Thanks!

The second part of the tutorial covering Bayesian optimization and Gaussian
processes can be found
[here](https://github.com/ritzann/2024-thrill-school-gp-bo).

# Contributing

Please see the [contributing guide](CONTRIBUTING.md). Make sure to only modify
or add the `.py` files, not the `.ipynb` files.

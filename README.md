# Teaching material for the course "Introduction to machine learning" at the 2024 THRILL summer school

https://indico.gsi.de/event/19869/

## Installation instructions for local execution

> [!IMPORTANT]
> Please install the software components as described below **before** the
> course and try to make sure that things work. If you run into problems, we
> will of course help you fix them on day 1 of course, but the goal is that
> most learners have their a software setup ready to go.


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

* clone this repo: `git clone https://github.com/elcorto/2024-thrill-school-machine-learning`
* change into the repo root folder: `cd 2024-thrill-school-machine-learning`
* (optional) install CPU-only [`torch`](https://pytorch.org/) (all code
  examples use small models and a toy dataset, so running on a laptop without a
  GPU is fine)

  ```sh
  uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
  ```

  or

  ```sh
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
  ```

* install all other requirements via

  ```sh
  uv pip install -r ./requirements.txt
  ```

  or

  ```sh
  python -m pip install -r ./requirements.txt
  ```


## Check your software stack

This repo has a small utility prepared which can check if you software
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

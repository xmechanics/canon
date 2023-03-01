# Canon #

Canon is a python package for X-Ray Laue diffractometer analysis.

Download and upzip the repository. Install the package by running the following in the unzipped folder:

    python setup.py install

Then import the library in your python script

    import canon

This is a small project maintained by only a tiny group,
therefore does not have enough human resource to keep up a comprehensive documentation.
Hopefully, the code is sufficiently self-explanatory.

## Development

1. Install [Miniconda](http://conda.pydata.org/miniconda.html).
2. Create conda env, run following command under project root
    ```shell
    conda env update -f conda-env.yaml -p ./venv
    ```
3. Activate it
    ```shell
    conda activate ./venv
    ```
4. While the virtual environment is activated, in `scripts` folder, copy `normalize_mpi.py` and modify to convert tiff files to jpg in parallel
    1. Put input tiff and output jpg in the `scripts/data` folder
5. Still in `scripts` folder, start jupyterlab
    ```shell
    jupyter-lab
    ```
    (The console output has a url like: http://localhost:8888/lab?token=blahblahbla, open that url in browser.)
6. In Jupyter Lab
    1. First, copy `extract_features.ipynb` notebook and modify. It converts jpg to feature metrics with selected models.
    2. Pick `models = ['ae_conv_4_256_best']`.
    3. After features matrices are generated and saved to disk, copy one of the `ex_...` notebooks and modify to analyze the features, with the chosen model.

## Dependencies

Recommand to use pre-compiled scientific python distributions, e.g. [Miniconda](http://conda.pydata.org/miniconda.html), a lightweight free version of [Anaconda](https://store.continuum.io/cshop/anaconda/).

### Required

- numpy
- scipy
- scikit-image

### Optional

- numba

## Licence

[The MIT Lincense](http://opensource.org/licenses/MIT)

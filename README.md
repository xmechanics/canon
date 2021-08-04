# Canon #

[![Travis Status](https://travis-ci.org/structrans/Canon.svg?branch=master)](https://travis-ci.org/structrans/Canon)
[![Coverage Status](https://coveralls.io/repos/structrans/Canon/badge.svg?branch=master&service=github)](https://coveralls.io/github/structrans/Canon?branch=master)
[![Codeship Status](https://codeship.com/projects/1dcd7cc0-0fe7-0133-d4b2-1e6fe7bb1028/status?branch=master)](https://codeship.com/projects/91981)

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
    conda env update -f conda.yaml
    ```
3. Activate it
    ```shell
    source activate canon
    ```
    or
    ```shell
    conda activate canon
    ```
4. While the `canon` conda environment is activated, in `scripts` folder, modify `normalize_mpi.py` to convert tiff files to jpg in parallel
5. Still in `scripts` folder, start jupyterlab
    ```shell
    jupyter-lab
    ```
    (The console output has a url like: http://localhost:8888/lab?token=blahblahbla, open that url in browser.)
6. In Jupyter Lab
    1. First, use `extract_features.ipynb` notebook. It converts jpg to feature metrics with selected models.
       Pick `models = ['ae_conv_4_256_best']`.
    2. After features matrices are generated and saved to disk, go to one of the `ex_...` notebook to analyze the features, wit the chosen model.

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

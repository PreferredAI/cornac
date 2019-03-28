Installation
============

Currently, we are supporting Python 3 (version 3.6 is recommended). There are several ways to install Cornac:

- **From PyPI (you may need a C++ compiler)**::

    pip3 install cornac

- **From Anaconda**::

    conda install cornac -c qttruong -c pytorch

- **From the GitHub source (for latest updates)**::

    pip3 install Cython
    git clone https://github.com/PreferredAI/cornac.git
    cd cornac
    python3 setup.py install

**Note:**

Additional dependencies required by models are listed here_.

.. _here: https://github.com/PreferredAI/cornac/blob/master/cornac/models#models

Some of the algorithms use `OpenMP` to support multi-threading. For OSX users, in order to run those algorithms efficiently, you might need to install `gcc` from Homebrew to have an OpenMP compiler::

    brew install gcc | brew link gcc

If you want to utilize your GPUs, you might consider:

- TensorFlow installation instructions: https://www.tensorflow.org/install/
- PyTorch installation instructions: https://pytorch.org/get-started/locally/
- cuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/ (for Nvidia GPUs)
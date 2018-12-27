Installation
=============

Currently, we are supporting Python 3 (version 3.6 is recommended). There are several ways to install Cornac:

- **From PyPI (you may need a C compiler)**::

	pip3 install cornac

- **From Anaconda**::

	conda install cornac -c qttruong -c pytorch

- **From the GitHub source (for latest updates)**::

	# Optional: install Cython
	pip3 install cython
	
	# Clone Cornac from the main repository
	git clone https://github.com/PreferredAI/cornac.git
	cd cornac
	
	# You will need a C compiler
	python3 setup.py install

**Note:** 

Some installed dependencies are CPU versions. If you want to utilize your GPU, you might consider:

- TensorFlow installation instructions: https://www.tensorflow.org/install/.
- PyTorch installation instructions: https://pytorch.org/get-started/locally/.
- cuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/ (for Nvidia GPUs).
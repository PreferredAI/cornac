# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

## Getting started

Getting started with Cornac is simple, and you just need to install it first.

### Installation

Please make sure you are using Python 3 (version >=3.6, is recommended), and you are on the latest pip.
Then, please run the appropriate Cornac install command according to your platform.

* **Windows**:
 
	- Some recommender models run with PyTorch. The latter library is not in PyPI, so when you install Cornac from the Wheel file this dependency is not handled automatically. You will, therefore need to install PyTorch first.::

	```
	pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl 
	pip install https://github.com/PreferredAI/cornac/raw/master/dist/cornac-0.1.0-cp36-cp36m-win_amd64.whl
	```

* **Linux**:
	```
	pip install https://github.com/PreferredAI/cornac/archive/master.zip --process-dependency-links
	```

* **MacOS**:
	```
	pip install https://github.com/PreferredAI/cornac/archive/master.zip
	```
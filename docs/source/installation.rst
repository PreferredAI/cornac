Installation
=============

Before to install Cornac, please make sure you are using Python 3 (version >=3.6, is recommended), and you are on the latest pip.
Then, please run the appropriate Cornac install command according to your platform.

* **Windows**:
 
	- Some recommender models run with PyTorch. The latter library is not in PyPI, so when you install Cornac from the Wheel file this dependency is not handled automatically. You will, therefore need to install PyTorch first.
	
 ::

	pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl 
	pip3 install https://github.com/PreferredAI/cornac/raw/master/dist/cornac-0.1.0-cp36-cp36m-win_amd64.whl

* **Linux**::

	pip3 install https://github.com/PreferredAI/cornac/archive/master.zip --process-dependency-links
	
* **MacOS**::

	pip3 install https://github.com/PreferredAI/cornac/archive/master.zip
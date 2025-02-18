# Dependencies
Training of SANSA uses [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), which depends on the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) numerical library. To install SuiteSparse on Ubuntu and macOS, run the commands below:
```
# Ubuntu
sudo apt-get install libsuitesparse-dev

# macOS
brew install suite-sparse
```
After installing SuiteSparse, simply install the requirements.txt.
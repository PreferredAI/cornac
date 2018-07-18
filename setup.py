
import setuptools
import os

	  
try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True	  
	  
	  
with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
#cython c++ modules
external_modules = []
if use_cython:
	ext_c2pf = setuptools.Extension("c2pf",
                  sources=["./cornac/models/context_cf/cython/c2pf.pyx", "./cornac/models/context_cf/cpp/cpp_c2pf.cpp"],
                  libraries=[],
                  include_dirs=['./cornac/models/context_cf/cpp/','./cornac/utils/external/eigen/Eigen','./cornac/utils/external/eigen/unsupported/Eigen/'],
                  language="c++" )
	external_modules += cythonize(ext_c2pf)
	#
	ext_pmf = './cornac/models/cf/cython/pmf.pyx'
	external_modules += cythonize(ext_pmf)
				  
				  

else:
    ext_c2pf = [setuptools.Extension('c2pf',
					['./cornac/models/context_cf/cython/c2pf.cpp'],
					include_dirs=['./cornac/models/context_cf/cpp/','./cornac/utils/external/eigen/Eigen','./cornac/utils/external/eigen/unsupported/Eigen/'],
					language="c++")]
    external_modules += ext_c2pf
    #
    ext_pmf = [setuptools.Extension('pmf',
                    ['./cornac/models/cf/cython/pmf.c'])]
    external_modules += ext_pmf
	
	
#Handling PyTorch dependency
if os.name == 'nt':
	torch_dl = 'http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl'
elif os.name == 'posix':
	torch_dl = 'http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl'	
	
	
setuptools.setup(
    name="cornac",
    version="0.1.0",
    author="Aghiles Salah",
    author_email="asalah@smu.edu.sg",
    description="A collection of recommendation algorithms and comparisons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    zip_safe=False,
    ext_modules = external_modules,
	install_requires=['numpy', 'scipy', 'pandas','torch==0.4.0'],
	dependency_links = [torch_dl],
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
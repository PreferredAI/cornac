import setuptools
import os


try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

with open('README.md', 'r') as fh:
    long_description = fh.read()

# cython c++ modules
external_modules = []
if use_cython:
    ext_c2pf = setuptools.Extension('c2pf',
                                    sources=[
                                        'cornac/models/c2pf/cython/c2pf.pyx',
                                        'cornac/models/c2pf/cpp/cpp_c2pf.cpp'],
                                    libraries=[],
                                    include_dirs=[
                                        'cornac/models/c2pf/cpp/',
                                        'cornac/utils/external/eigen/Eigen',
                                        'cornac/utils/external/eigen/unsupported/Eigen/'
                                    ],
                                    language='c++')
    external_modules += cythonize(ext_c2pf)
    #
    ext_pmf = 'cornac/models/pmf/cython/pmf.pyx'
    external_modules += cythonize(ext_pmf)
else:
    ext_c2pf = [setuptools.Extension('c2pf',
                                     sources=[
                                         'cornac/models/c2pf/cython/c2pf.cpp',
                                         'cornac/models/c2pf/cpp/cpp_c2pf.cpp'
                                     ],
                                     language='c++',
                                     include_dirs=[
                                         'cornac/models/c2pf/cpp/',
                                         'cornac/utils/external/eigen/Eigen',
                                         'cornac/utils/external/eigen/unsupported/Eigen/'
                                     ])]
    external_modules += ext_c2pf
    #
    ext_pmf = [setuptools.Extension('pmf',
                                    ['cornac/models/pmf/cython/pmf.c'])]
    external_modules += ext_pmf

# Handling PyTorch dependency
if os.name == 'nt':
    torch_dl = 'http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl'
elif os.name == 'posix':
    torch_dl = 'http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl'

setuptools.setup(
    name='cornac',
    version='0.1.0.post4',
    author='Aghiles Salah',
    author_email='asalah@smu.edu.sg',
    description='A collection of recommendation algorithms and comparisons',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://cornac.preferred.ai/',
    download_url='https://github.com/PreferredAI/cornac/archive/v0.1.0.tar.gz',
    keywords=['recommender', 'recommendation', 'factorization', 'multimodal'],
    zip_safe=False,
    ext_modules=external_modules,
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'tensorflow>=1.2.1',
        'torch>=0.4.1'
    ],
    dependency_links=[torch_dl],
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ),
)

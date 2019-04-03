from setuptools import Extension, setup, find_packages
import sys
import platform
import glob
import os

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

with open('README.md', 'r') as fh:
    long_description = fh.read()

USE_OPENMP = True


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']
    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


if sys.platform.startswith("win"):
    # compile args from
    # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
    compile_args = ['/O2', '/openmp']
    link_args = []
else:
    gcc = extract_gcc_binaries()
    if gcc is not None:
        rpath = '/usr/local/opt/gcc/lib/gcc/' + gcc[-1] + '/'
        link_args = ['-Wl,-rpath,' + rpath]
    else:
        link_args = []

    compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']

    if 'darwin' in platform.platform().lower():
        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc
        else:
            USE_OPENMP = False
            print('No GCC available. Install gcc from Homebrew '
                  'using brew install gcc.')
            # required arguments for default gcc of OSX
            compile_args.extend(['-O2', '-stdlib=libc++', '-mmacosx-version-min=10.7'])
            link_args.extend(['-O2', '-stdlib=libc++', '-mmacosx-version-min=10.7'])

    if USE_OPENMP:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension(name='cornac.models.c2pf.c2pf',
              sources=[
                  'cornac/models/c2pf/cython/c2pf' + ext,
                  'cornac/models/c2pf/cpp/cpp_c2pf.cpp'],
              include_dirs=[
                  'cornac/models/c2pf/cpp/',
                  'cornac/utils/external/eigen/Eigen',
                  'cornac/utils/external/eigen/unsupported/Eigen/'
              ],
              language='c++'),
    Extension(name='cornac.models.pmf.pmf',
              sources=['cornac/models/pmf/cython/pmf' + ext],
              language='c++'),
    Extension(name='cornac.models.mcf.mcf',
              sources=['cornac/models/mcf/cython/mcf' + ext],
              language='c++'),
    Extension('cornac.models.hpf.hpf',
              sources=['cornac/models/hpf/cython/hpf' + ext,
                       'cornac/models/hpf/cpp/cpp_hpf.cpp'],
              include_dirs=[
                  'cornac/models/hpf/cpp/',
                  'cornac/utils/external/eigen/Eigen',
                  'cornac/utils/external/eigen/unsupported/Eigen/'
              ],
              language='c++'),
    Extension(name='cornac.models.mf.recom_mf',
              sources=['cornac/models/mf/recom_mf' + ext],
              language='c++'),
    Extension(name='cornac.models.bpr.recom_bpr',
              sources=['cornac/models/bpr/recom_bpr' + ext],
              language='c++',
              extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='cornac.utils.fast_dot',
              sources=['cornac/utils/fast_dot' + ext],
              language='c++'),
]

cmdclass = {}

# cythonize c++ modules
if USE_CYTHON:
    extensions = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})

setup(
    name='cornac',
    version='0.1.0.post5',
    author='Aghiles Salah',
    author_email='asalah@smu.edu.sg',
    description='A collection of recommendation algorithms and comparisons',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://cornac.preferred.ai/',
    keywords=['recommender', 'recommendation', 'factorization', 'multimodal'],
    ext_modules=extensions,
    install_requires=[
        'numpy',
        'scipy',
        'tqdm>=4.19'
    ],
    extras_require={
        'tests': ['pytest',
                  'pytest-pep8',
                  'pytest-xdist',
                  'pytest-cov']
    },
    cmdclass=cmdclass,
    packages=find_packages(),
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

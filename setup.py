# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import glob
from setuptools import Extension, setup, find_packages


"""
Release instruction:
    - Check that tests run correctly with all CI tools.
    - Change __version__ in setup.py, cornac/__init__.py, docs/source/conf.py.
    - Commit and release a version on GitHub, Actions will be triggered to build and upload to PyPI.
    - Update conda-forge feedstock with new version and SHA256 hash of the new .tar.gz archive on PyPI (optional), the conda-forge bot will detect a new version and create PR after a while.
    - Check on https://anaconda.org/conda-forge/cornac that new version is available for all platforms.
"""


try:
    import numpy as np
except ImportError:
    exit("Please install numpy>=1.14 first.")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True


with open("README.md", "r") as fh:
    long_description = fh.read()


USE_OPENMP = True


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9].[0-9]",
        "/opt/local/bin/g++-mp-[0-9]",
        "/usr/local/bin/g++-[0-9].[0-9]",
        "/usr/local/bin/g++-[0-9]",
    ]
    if sys.platform.startswith("darwin"):
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
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    gcc = extract_gcc_binaries()
    if gcc is not None:
        rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
        link_args = ["-Wl,-rpath," + rpath]
    else:
        link_args = []

    compile_args = [
        "-Wno-unused-function",
        "-Wno-maybe-uninitialized",
        "-O3",
        "-ffast-math",
    ]

    if sys.platform.startswith("darwin"):
        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc
        else:
            USE_OPENMP = False
            print("No GCC available. Install gcc from Homebrew using brew install gcc.")
            # required arguments for default gcc of OSX
            compile_args.extend(["-O2", "-stdlib=libc++", "-mmacosx-version-min=10.7"])
            link_args.extend(["-O2", "-stdlib=libc++", "-mmacosx-version-min=10.7"])
    
    if USE_OPENMP:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")

ext = ".pyx" if USE_CYTHON else ".cpp"

extensions = [
    Extension(
        name="cornac.models.c2pf.c2pf",
        sources=[
            "cornac/models/c2pf/cython/c2pf" + ext,
            "cornac/models/c2pf/cpp/cpp_c2pf.cpp",
        ],
        include_dirs=[
            "cornac/models/c2pf/cpp/",
            "cornac/utils/external/eigen/Eigen",
            "cornac/utils/external/eigen/unsupported/Eigen/",
        ],
        language="c++",
    ),
    Extension(
        name="cornac.models.nmf.recom_nmf",
        sources=["cornac/models/nmf/recom_nmf" + ext],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.pmf.pmf",
        sources=["cornac/models/pmf/cython/pmf" + ext],
        language="c++",
    ),
    Extension(
        name="cornac.models.mcf.mcf",
        sources=["cornac/models/mcf/cython/mcf" + ext],
        language="c++",
    ),
    Extension(
        name="cornac.models.sorec.sorec",
        sources=["cornac/models/sorec/cython/sorec" + ext],
        language="c++",
    ),
    Extension(
        "cornac.models.hpf.hpf",
        sources=["cornac/models/hpf/cython/hpf" + ext, "cornac/models/hpf/cpp/cpp_hpf.cpp",],
        include_dirs=[
            "cornac/models/hpf/cpp/",
            "cornac/utils/external/eigen/Eigen",
            "cornac/utils/external/eigen/unsupported/Eigen/",
        ],
        language="c++",
    ),
    Extension(
        name="cornac.models.mf.recom_mf",
        sources=["cornac/models/mf/recom_mf" + ext],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.baseline_only.recom_bo",
        sources=["cornac/models/baseline_only/recom_bo" + ext],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.efm.recom_efm",
        sources=["cornac/models/efm/recom_efm" + ext],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.comparer.recom_comparer_obj",
        sources=["cornac/models/comparer/recom_comparer_obj" + ext],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.bpr.recom_bpr",
        sources=["cornac/models/bpr/recom_bpr" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.bpr.recom_wbpr",
        sources=["cornac/models/bpr/recom_wbpr" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.sbpr.recom_sbpr",
        sources=["cornac/models/sbpr/recom_sbpr" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.mter.recom_mter",
        sources=["cornac/models/mter/recom_mter" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.comparer.recom_comparer_sub",
        sources=["cornac/models/comparer/recom_comparer_sub" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.mmmf.recom_mmmf",
        sources=["cornac/models/mmmf/recom_mmmf" + ext],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.knn.similarity",
        sources=["cornac/models/knn/similarity" + ext],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.utils.fast_dict",
        sources=["cornac/utils/fast_dict" + ext],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.utils.fast_dot",
        sources=["cornac/utils/fast_dot" + ext],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.utils.fast_sparse_funcs",
        sources=["cornac/utils/fast_sparse_funcs" + ext],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

if sys.platform.startswith("linux"):  # Linux supported only
    extensions += [
        Extension(
            name="cornac.models.fm.recom_fm",
            sources=["cornac/models/fm/recom_fm" + ext],
            include_dirs=[
                np.get_include(),
                "cornac/models/fm/libfm/util",
                "cornac/models/fm/libfm/fm_core/",
                "cornac/models/fm/libfm/libfm/src/",
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]

cmdclass = {}

# cythonize c++ modules
if USE_CYTHON:
    extensions = cythonize(extensions)
    cmdclass.update({"build_ext": build_ext})

setup(
    name="cornac",
    version="1.13.5",
    description="A Comparative Framework for Multimodal Recommender Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cornac.preferred.ai",
    keywords=[
        "recommender system",
        "collaborative filtering",
        "multimodal",
        "preference learning",
        "recommendation",
    ],
    ext_modules=extensions,
    install_requires=["numpy", "scipy", "tqdm>=4.19", "powerlaw"],
    extras_require={"tests": ["pytest", "pytest-pep8", "pytest-xdist", "pytest-cov"]},
    cmdclass=cmdclass,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
)

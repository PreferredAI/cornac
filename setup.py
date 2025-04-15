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


"""
Release instruction:
    - Check that tests run correctly with all CI tools.
    - Change __version__ in pyproject.toml, cornac/__init__.py, docs/source/conf.py.
    - Commit and release a version on GitHub, Actions will be triggered to build and upload to PyPI.
    - Update conda-forge feedstock with new version and SHA256 hash of the new .tar.gz archive on PyPI (optional), the conda-forge bot will detect a new version and create PR after a while.
    - Check on https://anaconda.org/conda-forge/cornac that new version is available for all platforms.
"""


import os
import sys
import glob
import shutil
from setuptools import Extension, Command, setup, find_packages
from Cython.Distutils import build_ext
import numpy as np


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
            if not os.path.exists("/usr/bin/g++"):
                print(
                    "No GCC available. Install gcc from Homebrew using brew install gcc."
                )
            USE_OPENMP = False
            # required arguments for default gcc of OSX
            compile_args.extend(["-O2", "-stdlib=libc++", "-mmacosx-version-min=10.7"])
            link_args.extend(["-O2", "-stdlib=libc++", "-mmacosx-version-min=10.7"])

    if USE_OPENMP:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")


extensions = [
    Extension(
        name="cornac.models.c2pf.c2pf",
        sources=[
            "cornac/models/c2pf/cython/c2pf.pyx",
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
        sources=["cornac/models/nmf/recom_nmf.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.pmf.pmf",
        sources=["cornac/models/pmf/cython/pmf.pyx"],
        language="c++",
    ),
    Extension(
        name="cornac.models.mcf.mcf",
        sources=["cornac/models/mcf/cython/mcf.pyx"],
        language="c++",
    ),
    Extension(
        name="cornac.models.sorec.sorec",
        sources=["cornac/models/sorec/cython/sorec.pyx"],
        language="c++",
    ),
    Extension(
        "cornac.models.hpf.hpf",
        sources=[
            "cornac/models/hpf/cython/hpf.pyx",
            "cornac/models/hpf/cpp/cpp_hpf.cpp",
        ],
        include_dirs=[
            "cornac/models/hpf/cpp/",
            "cornac/utils/external/eigen/Eigen",
            "cornac/utils/external/eigen/unsupported/Eigen/",
        ],
        language="c++",
    ),
    Extension(
        name="cornac.models.mf.backend_cpu",
        sources=["cornac/models/mf/backend_cpu.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.baseline_only.recom_bo",
        sources=["cornac/models/baseline_only/recom_bo.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.efm.recom_efm",
        sources=["cornac/models/efm/recom_efm.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.comparer.recom_comparer_obj",
        sources=["cornac/models/comparer/recom_comparer_obj.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.models.bpr.recom_bpr",
        sources=["cornac/models/bpr/recom_bpr.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.bpr.recom_wbpr",
        sources=["cornac/models/bpr/recom_wbpr.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.sbpr.recom_sbpr",
        sources=["cornac/models/sbpr/recom_sbpr.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.lrppm.recom_lrppm",
        sources=["cornac/models/lrppm/recom_lrppm.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.mter.recom_mter",
        sources=["cornac/models/mter/recom_mter.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.companion.recom_companion",
        sources=["cornac/models/companion/recom_companion.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.comparer.recom_comparer_sub",
        sources=["cornac/models/comparer/recom_comparer_sub.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.mmmf.recom_mmmf",
        sources=["cornac/models/mmmf/recom_mmmf.pyx"],
        include_dirs=[np.get_include(), "cornac/utils/external"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.knn.similarity",
        sources=["cornac/models/knn/similarity.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.utils.fast_dict",
        sources=["cornac/utils/fast_dict.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        name="cornac.utils.fast_dot",
        sources=["cornac/utils/fast_dot.pyx"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.utils.fast_sparse_funcs",
        sources=["cornac/utils/fast_sparse_funcs.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

if sys.platform.startswith("linux"):  # Linux supported only
    extensions += [
        Extension(
            name="cornac.models.fm.backend_libfm",
            sources=["cornac/models/fm/backend_libfm.pyx"],
            include_dirs=[
                np.get_include(),
                "cornac/models/fm/libfm/util/",
                "cornac/models/fm/libfm/fm_core/",
                "cornac/models/fm/libfm/libfm/src/",
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove .cpp and .so files for a clean build
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("cornac"):
            for filename in filenames:
                root, extension = os.path.splitext(filename)

                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                if extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {
    "clean": CleanCommand,
    "build_ext": build_ext,
}

setup(
    ext_modules=extensions,
    extras_require={"tests": ["pytest", "pytest-pep8", "pytest-xdist", "pytest-cov", "Flask"]},
    cmdclass=cmdclass,
    packages=find_packages(),
)

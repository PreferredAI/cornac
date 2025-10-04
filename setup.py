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
import platform, re
from setuptools import Extension, Command, setup, find_packages
from Cython.Distutils import build_ext
import numpy as np


with open("README.md", "r") as fh:
    long_description = fh.read()


USE_OPENMP = True

def homebrew_prefix():
    # Prefer explicit env, then sane defaults for Intel vs Apple silicon
    return os.environ.get(
        "HOMEBREW_PREFIX",
        "/opt/homebrew" if platform.machine() == "arm64" else "/usr/local"
    )

def extract_gcc_binary():
    # Return full path to latest g++-NN from Homebrew or MacPorts
    patterns = [
        f"{homebrew_prefix()}/bin/g++-[0-9]*",      # Homebrew (both /usr/local and /opt/homebrew)
        "/usr/local/bin/g++-[0-9]*",                # Legacy Intel HB
        "/opt/local/bin/g++-mp-[0-9]*",             # MacPorts
    ]
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands:
        return None
    cands.sort()
    return cands[-1]  # newest version string-wise

if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = [
        "-Wno-unused-function", 
        "-Wno-maybe-uninitialized",
        "-O3", 
        "-ffast-math"
    ]
    link_args = []

    if sys.platform.startswith("darwin"):
        gcc_path = extract_gcc_binary()

        # Choose a valid deployment target
        if platform.machine() == "arm64":
            mac_min = "11.0"  # arm64 cannot target < 11.0
        else:
            mac_min = "10.13"  # bump from 10.7; keep reasonably old but supported

        compile_args.extend(["-stdlib=libc++", f"-mmacosx-version-min={mac_min}"]) 
        link_args.extend(["-stdlib=libc++", f"-mmacosx-version-min={mac_min}"])

        if gcc_path is not None:
            # Use Homebrew/MacPorts GCC for OpenMP (libgomp)
            os.environ["CC"] = gcc_path
            os.environ["CXX"] = gcc_path

            # rpath to GCC’s libgomp dir
            prefix = os.path.dirname(os.path.dirname(gcc_path))  # .../bin -> prefix
            # For Homebrew GCC the libs live under <prefix>/opt/gcc/lib/gcc/<MAJOR>
            # Try to extract MAJOR from the binary name (g++-14, g++-13, etc.)
            m = re.search(r'(\d+)(?:\.\d+)?$', os.path.basename(gcc_path))
            gcc_major = m.group(1) if m else ""
            hb_opt_gcc = f"{homebrew_prefix()}/opt/gcc/lib/gcc/{gcc_major}"
            mp_libgcc = "/opt/local/lib/gcc{}".format(gcc_major) if prefix.startswith("/opt/local") else None

            if os.path.isdir(hb_opt_gcc):
                link_args.append(f"-Wl,-rpath,{hb_opt_gcc}")
            elif mp_libgcc and os.path.isdir(mp_libgcc):
                link_args.append(f"-Wl,-rpath,{mp_libgcc}")

            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")
            # Deployment target is still needed for ABI consistency
            compile_args.extend(["-stdlib=libc++", f"-mmacosx-version-min={mac_min}"])
            link_args.extend(["-stdlib=libc++", f"-mmacosx-version-min={mac_min}"])
        else:
            # No GCC found → default to Apple clang. Either disable OpenMP or use libomp if present.
            USE_OPENMP = False
            compile_args.extend(["-O2", "-stdlib=libc++", f"-mmacosx-version-min={mac_min}"])
            link_args.extend(["-O2", "-stdlib=libc++", f"-mmacosx-version-min={mac_min}"])

            # Optional: enable OpenMP with clang + libomp if installed
            hb = homebrew_prefix()
            omp_inc = f"{hb}/opt/libomp/include"
            omp_lib = f"{hb}/opt/libomp/lib"
            if os.path.exists(os.path.join(omp_inc, "omp.h")) and os.path.isdir(omp_lib):
                # clang needs -Xpreprocessor -fopenmp and links against -lomp
                compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_inc}"]
                link_args += [f"-L{omp_lib}", "-lomp"]
                USE_OPENMP = True

    # Common C++ standard
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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.nmf.recom_nmf",
        sources=["cornac/models/nmf/recom_nmf.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.pmf.pmf",
        sources=["cornac/models/pmf/cython/pmf.pyx"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.mcf.mcf",
        sources=["cornac/models/mcf/cython/mcf.pyx"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.sorec.sorec",
        sources=["cornac/models/sorec/cython/sorec.pyx"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="cornac.models.comparer.recom_comparer_obj",
        sources=["cornac/models/comparer/recom_comparer_obj.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
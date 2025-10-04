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
import shutil
import platform
from setuptools import Extension, Command, setup, find_packages
from Cython.Distutils import build_ext
import numpy as np


with open("README.md", "r") as fh:
    long_description = fh.read()


USE_OPENMP = False  # we'll turn it on only if we find a working libomp

def candidates_from_env():
    # Let users/CI point to a custom libomp
    incs = []
    libs = []
    omp_dir = os.environ.get("OMP_DIR")
    if omp_dir:
        incs += [os.path.join(omp_dir, "include")]
        libs += [os.path.join(omp_dir, "lib")]
    inc_env = os.environ.get("OMP_INCLUDE")
    lib_env = os.environ.get("OMP_LIB")
    if inc_env: incs.append(inc_env)
    if lib_env: libs.append(lib_env)
    # Conda environments
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        incs += [os.path.join(conda, "include")]
        libs += [os.path.join(conda, "lib")]
    # Homebrew (Apple silicon and Intel) and MacPorts (optional, don’t require them)
    incs += ["/opt/homebrew/opt/libomp/include", "/usr/local/opt/libomp/include", "/opt/local/include"]
    libs += ["/opt/homebrew/opt/libomp/lib",     "/usr/local/opt/libomp/lib",     "/opt/local/lib"]
    # Common fallbacks
    incs += ["/usr/local/include", "/usr/include"]
    libs += ["/usr/local/lib", "/usr/lib"]
    # De-dup while keeping order
    def uniq(xs):
        seen=set(); out=[]
        for x in xs:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out
    return uniq(incs), uniq(libs)

def find_libomp():
    incs, libs = candidates_from_env()
    inc_dir = next((d for d in incs if os.path.exists(os.path.join(d, "omp.h"))), None)
    # Prefer a real libomp over stubs
    lib_names = ["libomp.dylib", "libomp.a"]
    lib_dir = None
    for d in libs:
        if any(os.path.exists(os.path.join(d, n)) for n in lib_names):
            lib_dir = d
            break
    return inc_dir, lib_dir

compile_args = []
link_args = []

if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
elif sys.platform.startswith("darwin"):
    # Always use Clang on macOS
    os.environ.setdefault("CC", "clang")
    os.environ.setdefault("CXX", "clang++")

    # Force single-arch arm64 on Apple silicon unless caller overrides
    if platform.machine() == "arm64" and not os.environ.get("ARCHFLAGS"):
        os.environ["ARCHFLAGS"] = "-arch arm64"

    mac_min = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "12.0")

    # Base flags good for Clang/libc++
    compile_args += [
        "-O3", "-ffast-math",
        "-Wno-unused-function",
        "-std=c++11",
        "-stdlib=libc++",
        f"-mmacosx-version-min={mac_min}",
    ]
    link_args += [
        "-std=c++11",
        "-stdlib=libc++",
        f"-mmacosx-version-min={mac_min}",
    ]

    # Optional OpenMP (only if a usable libomp is present)
    if os.environ.get("CORNAC_DISABLE_OPENMP") != "1":
        inc_dir, lib_dir = find_libomp()
        if inc_dir and lib_dir:
            compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{inc_dir}"]
            link_args += [f"-L{lib_dir}", "-lomp", f"-Wl,-rpath,{lib_dir}"]
            USE_OPENMP = True
        elif os.environ.get("CORNAC_FORCE_OPENMP") == "1":
            raise RuntimeError(
                "CORNAC_FORCE_OPENMP=1 but libomp was not found; set OMP_INCLUDE/OMP_LIB or OMP_DIR."
            )
else:
    # Linux/Unix: prefer OpenMP via compiler default (GCC/Clang + libgomp)
    compile_args += [
        "-O3", "-ffast-math",
        "-Wno-unused-function",
        "-Wno-maybe-uninitialized",
        "-std=c++11",
        "-fopenmp",
    ]
    link_args += ["-std=c++11", "-fopenmp"]
    USE_OPENMP = True


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
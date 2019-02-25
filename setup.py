from setuptools import Extension, setup, find_packages
import numpy

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

with open('README.md', 'r') as fh:
    long_description = fh.read()

ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension(name='c2pf',
              sources=[
                  'cornac/models/c2pf/cython/c2pf' + ext,
                  'cornac/models/c2pf/cpp/cpp_c2pf.cpp'],
              include_dirs=[
                  'cornac/models/c2pf/cpp/',
                  'cornac/utils/external/eigen/Eigen',
                  'cornac/utils/external/eigen/unsupported/Eigen/'
              ],
              language='c++'),
    Extension(name='pmf',
              sources=['cornac/models/pmf/cython/pmf' + ext],
              language='c++'),
    Extension('hpf',
              sources=['cornac/models/hpf/cython/hpf' + ext,
                       'cornac/models/hpf/cpp/cpp_hpf.cpp'],
              include_dirs=[
                  'cornac/models/hpf/cpp/',
                  'cornac/utils/external/eigen/Eigen',
                  'cornac/utils/external/eigen/unsupported/Eigen/'
              ],
              language='c++'),
    Extension(name='recom_mf',
              sources=['cornac/models/mf/recom_mf' + ext],
              include_dirs=[numpy.get_include()],
              language='c++')
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
    download_url='https://github.com/PreferredAI/cornac/archive/v0.1.0.tar.gz',
    keywords=['recommender', 'recommendation', 'factorization', 'multimodal'],
    ext_modules=extensions,
    install_requires=[
        'numpy',
        'scipy',
        'pandas'
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

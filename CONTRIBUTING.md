# Contributing to Cornac

Cornac is an open-source project and we welcome contributions of all kinds.
We welcome pull requests for new features such as models, datasets, metrics
and anything that could make Cornac a better tool for all.

This page sums up what we check when we review a pull request. The [Contributors Guide](https://cornac.readthedocs.io/en/stable/developer/index.html) walks you through forking the repository and opening a pull request, and the tutorials show how to [add a model](https://github.com/PreferredAI/cornac/blob/master/tutorials/add_model.md) and [add a metric](https://github.com/PreferredAI/cornac/blob/master/tutorials/add_metric.md).

## Development Setup

Cornac needs Python 3.10 or newer and a C++ compiler, because parts of it are written in Cython. In a clone of your fork, run:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
pip install Cython numpy scipy
python setup.py build_ext --inplace
pip install -e ".[tests]"
pytest
```

After you change a Cython (`.pyx`) file, run `python setup.py build_ext --inplace` again.

## Code Style

Cornac uses [Ruff](https://docs.astral.sh/ruff/) to check and format code, with its settings in `pyproject.toml`. You can install it with `pip install ruff`.

Check and format each new Python file by name:

```bash
ruff check path/to/new_file.py
ruff format path/to/new_file.py
```

Many existing files are not formatted with Ruff yet, so running `ruff format` on a whole existing file or folder would rewrite code you did not touch. In an existing file, format only the lines you changed, for example lines 120 to 135 with `ruff format --range=120-135 path/to/file.py`, and keep formatting changes out of functional changes, so that reviewers can see what your change does.

Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style, with `Parameters` and `Returns` sections, because the API reference is generated from them. New source files start with the Apache-2.0 license header; copy it from an existing file.

## Optional Dependencies

A core install of Cornac needs only `numpy`, `scipy`, `tqdm` and `powerlaw`, and users install libraries such as PyTorch or TensorFlow only for the models they use. Since `import cornac` loads the `recom_<name>.py` file of every model, these files import such libraries inside `fit()`, `score()` or other methods rather than at the top. Backend modules that are only imported from inside those methods can import them at the top as usual. Shared modules such as `cornac.data`, `cornac.eval_methods` and `cornac.metrics` import only the core dependencies.

List the extra packages your model needs in its own `requirements.txt`, not in `pyproject.toml`. To check that `import cornac` still loads neither PyTorch nor TensorFlow, run:

```bash
python -c "import sys, cornac; assert not {'torch', 'tensorflow'} & set(sys.modules)"
```

## Adding a Model, Dataset or Metric

A new model lives in its own package, `cornac/models/<name>/`, with an `__init__.py`, a `recom_<name>.py` that holds the recommender class, any backend modules, and a `requirements.txt` if it needs extra packages. Along with the package:

- export the class in `cornac/models/__init__.py`;
- add a row to the model table in `README.md`, following the layout of the existing rows, because the model page of the documentation is generated from it;
- add an entry to `docs/source/api_ref/models.rst`;
- add an example script under `examples/` and list it in `examples/README.md`.

A dataset loader is a module in `cornac/datasets/`. Add a test for it in `tests/cornac/datasets/`, an entry in `docs/source/api_ref/datasets.rst`, and a description in `cornac/datasets/README.md`.

A metric goes in `cornac/metrics/`, with tests in `tests/cornac/metrics/` and an entry in `docs/source/api_ref/metrics.rst`.

## Tests

Tests live under `tests/cornac/` and run with `pytest`. Changes to shared code, such as data handling, evaluation methods, metrics, experiments and the model base classes, should come with tests, because many models rely on that code. A new model does not need its own test suite, but it should come with an example script that runs, because the example is how reviewers and users check that the model works.

A plain `pytest` does not run the tests in `tests/cornac/datasets/`, because they download the datasets. To run them, name the folder or a file, for example `pytest tests/cornac/datasets/test_movielens.py`.

## Pull Requests

Open your pull request against the `master` branch and fill in the template.

Your pull request will be reviewed by Cornac maintainers.
They will provide feedback and request changes if necessary.
As this is an open source project, the repository is maintained on a voluntary
basis. We thank you for your patience during the review process.

## AI-Assisted Contributions

You are welcome to use AI tools, including coding agents, to help with your contribution. You are still the author, so you are responsible for the whole change.

Please read and understand every change before you open a pull request, and be ready to explain it during review. Report only test results that you ran yourself. If an AI tool wrote a large part of the code, tests or description, say so in the pull request and name the tool, so that reviewers know where to look closely. You can also add an `Assisted-by: <tool>` line to your commit messages.

Keep pull requests focused. Our maintainers review in their spare time, and we may close very large pull requests that their author does not seem to have reviewed. The same goes for issues and replies to review comments: please read what an AI tool wrote before you post it.

## License

By contributing to Cornac, you agree that your code will be released under the Apache 2.0 License.

Thank you for contributing to Cornac!

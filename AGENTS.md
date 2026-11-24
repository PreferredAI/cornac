# AGENTS.md

Notes for coding agents working on Cornac. [CONTRIBUTING.md](CONTRIBUTING.md) holds the rules for code style, optional dependencies, new models, tests and pull requests. Read it before you change code.

## AI-assisted contributions

The person who opens a pull request is its author, and reads and understands every change in it. Say in the pull request that an AI tool was used, and name the tool. Do not open issues or pull requests, or post replies, until a person has reviewed them.

## Commands

In a new Python environment, from the repository root:

```bash
pip install Cython numpy scipy
python setup.py build_ext --inplace
pip install -e ".[tests]"
pip install ruff
pytest tests/cornac/eval_methods/test_ratio_split.py  # the test file for your change
pytest                                                # the whole suite
```

Before opening a pull request:

```bash
python docs/check_listings.py
python -c "import sys, cornac; assert not {'torch', 'tensorflow'} & set(sys.modules)"
ruff check path/to/new_file.py
ruff format path/to/new_file.py
```

In existing files, format only the lines you changed, as [Code Style](CONTRIBUTING.md#code-style) shows.

## Before you commit

- Keep the pull request to the change that was asked for: no planning notes, notebooks, scratch scripts, debug code, unrelated fixes or reformatting.
- Look at what you commit. Tests leave a `saved_models/` folder in the repository root, and a docs build rewrites `docs/source/_static/models/data.js`; neither belongs in the pull request. `*.csv`, `*.zip` and `*.pickle` files are git-ignored, so a new data file in one of these formats is left out without a warning.
- Report only the tests and examples you ran, and claim no more than their output shows.

[Models and Tasks](docs/source/user/tasks.rst) describes the task types and the base class, dataset and evaluation method each one uses, and [tutorials/add_model.md](tutorials/add_model.md) walks through adding a model.

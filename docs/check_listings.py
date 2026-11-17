# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
"""Check that every model, dataset loader and example is listed in the docs.

Usage: python docs/check_listings.py
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def relative_imports(path):
    """Return (module, name) for each top-level `from .module import name`."""
    return [
        (node.module, alias.name)
        for node in ast.parse(read(path)).body
        if isinstance(node, ast.ImportFrom) and node.level == 1
        for alias in node.names
    ]


def autodoc_targets(path):
    return set(re.findall(r"^\.\. auto(?:module|class):: (\S+)", read(path), re.M))


def check_models():
    api_ref = autodoc_targets("docs/source/api_ref/models.rst")
    readme = read("README.md")
    for package, name in relative_imports("cornac/models/__init__.py"):
        if package == "recommender":
            continue
        modules = {
            n: m for m, n in relative_imports(f"cornac/models/{package}/__init__.py")
        }
        module = f"cornac.models.{package}.{modules[name]}"
        if module not in api_ref and f"{module}.{name}" not in api_ref:
            yield f"{name} is missing from docs/source/api_ref/models.rst"
        if (
            f"(cornac/models/{package})" not in readme
            and f"({module.replace('.', '/')}.py)" not in readme
        ):
            yield f"{name} is missing from the models table in README.md"


def check_datasets():
    api_ref = autodoc_targets("docs/source/api_ref/datasets.rst")
    readme = read("cornac/datasets/README.md")
    for path in sorted((ROOT / "cornac" / "datasets").glob("*.py")):
        if path.stem == "__init__":
            continue
        module = f"cornac.datasets.{path.stem}"
        if module not in api_ref:
            yield f"{module} is missing from docs/source/api_ref/datasets.rst"
        if not (
            re.search(rf"{re.escape(module)}\b", readme) or f"(./{path.name})" in readme
        ):
            yield f"{module} is missing from cornac/datasets/README.md"


def check_examples():
    readme = read("examples/README.md")
    for path in sorted((ROOT / "examples").iterdir()):
        if path.suffix in {".py", ".ipynb"} and f"]({path.name})" not in readme:
            yield f"{path.name} is missing from examples/README.md"


if __name__ == "__main__":
    problems = [*check_models(), *check_datasets(), *check_examples()]
    for problem in problems:
        print(problem)
    sys.exit(1 if problems else 0)

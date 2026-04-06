# New Project Setup Guideline

## Package Layout

Use `src/<package_name>/` as the source root, configured via `tool.setuptools.packages.find.where = ["src"]` in `pyproject.toml`.

Standard submodules:

```
src/<package_name>/
├── __init__.py
├── version.py
├── configs/
├── db/
│   └── models.py
├── entrypoints/
│   └── api.py
├── routers/
├── tasks/
├── types/
└── utils/
tests/
```

## pyproject.toml

Single source of truth for build config, linting, and testing.

### Build

- Build system: setuptools
- Version: pulled dynamically from `<package>.version.__version__`

```toml
[build-system]
requires = ["setuptools>=62.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.dynamic]
version = { attr = "<package>.version.__version__" }

[tool.setuptools.packages.find]
where = ["src"]
```

### Ruff

- Line length: 99
- Target: py312
- isort: package as first-party
- flake8-bugbear rules enabled, `E501` ignored

```toml
[tool.ruff]
line-length = 99
indent-width = 4
target-version = "py312"
extend-include = [".pyi?$", ".ipynb"]
extend-exclude = ["archive/*"]
respect-gitignore = true

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

[tool.ruff.lint]
select = ["E1", "E4", "E7", "E9", "F", "I", "W1", "W2", "W3", "W6", "B"]
ignore = ["E501"]
unfixable = ["B"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402"]
"**/{tests,docs,tools}/*" = ["E402"]

[tool.ruff.lint.isort]
known-first-party = ["<package>"]

[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = [
    "fastapi.Depends",
    "fastapi.File",
    "fastapi.Form",
    "fastapi.Path",
    "fastapi.Query",
]
```

### Pytest

- Timeout: 90s
- Asyncio mode: auto
- Test directory: `tests/`

```toml
[tool.pytest.ini_options]
timeout = 90
log_cli = true
asyncio_mode = "auto"
addopts = "--doctest-modules -vv -ra --strict-markers --durations=200 --durations-min=0.05"
testpaths = ["tests"]
```

### Optional Dependencies

Grouped as `lint`, `test`, and `all`:

```toml
[project.optional-dependencies]
lint = ["ruff~=0.12"]
test = [
    "pytest~=8.3",
    "pytest-asyncio>=0.23.8",
    "pytest-timeout~=2.3",
]
all = [
    "<package>[lint,test]",
]
```

## GitHub Actions Lint Workflow

File: `.github/workflows/lint.yml`

Triggered on push to any branch. Two jobs:

- **python_lint**: Install `ruff~=0.12`, run `ruff check --output-format github .` and `ruff format --diff .`
- **prettier_lint**: Run `npx prettier@3.8 --check .`

```yaml
name: Linting Workflow

on:
  push:
    branches:
      - "**"

jobs:
  python_lint:
    name: Python linting and formatting checks
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - name: Checkout code
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install linting libraries
        run: |
          python3 -m pip install "ruff~=0.12"

      - name: Check Python files using Ruff
        run: |
          ruff check --output-format github .
          ruff format --diff .

  prettier_lint:
    name: Prettier Checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v6

      - name: Setup Node.js
        uses: actions/setup-node@v6
        with:
          node-version: 24

      - name: Check files using Prettier
        run: npx prettier@3.8 --check .
```

## Versioning

`src/<package>/version.py` with:

```python
__version__ = "0.0.1"
```

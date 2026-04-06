# New Project Setup Guideline

## Package Layout

Use `src/<package>/` as the source root, configured via `tool.setuptools.packages.find.where = ["src"]` in `pyproject.toml`.

Standard submodules:

```
src/<package>/
├── __init__.py
├── configs.py
├── version.py
├── db/
│   └── orm.py
├── entrypoints/
│   └── api.py
├── routers/
├── types/
└── utils/
tests/
```

## pyproject.toml

Single source of truth for build config, linting, and testing.

### Build

- Build system: setuptools
- Version: pulled dynamically from `src/<package>/version.py` (`__version__ = "0.0.1"`)

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

## .gitattributes

Enforce consistent line endings and mark binary files so they aren't diffed or corrupted by EOL conversion.

- Default all files to `text=auto eol=lf`
- `.bat` files use CRLF; `.sh` files use LF
- Explicitly mark common text extensions (`.css`, `.html`, `.js*`, `.md`, `.py`, `.sh`) as `text`
- Mark media, archive, and data files as `binary`

```gitattributes
# Default: auto-detect text files and normalise to LF
* text=auto eol=lf

# Platform-specific EOL overrides
*.bat text eol=crlf
*.sh text eol=lf

# Explicit text files (normalise CRLF => LF)
*.css text
*.html text
*.js* text
*.md text
*.py text

# Binary files (no EOL conversion, no diff)
*.7z binary
*.bin binary
*.cu.o binary
*.db binary
*.doc binary
*.docx binary
*.gif binary
*.gz binary
*.heic* binary
*.heif* binary
*.jar binary
*.jpeg binary
*.jpg binary
*.mov binary
*.mp* binary
*.npy binary
*.npz binary
*.parquet binary
*.pcd binary
*.pdf binary
*.pkl binary
*.png binary
*.ppt binary
*.pptx binary
*.pth binary
*.so binary
*.webp binary
*.xls binary
*.xlsx binary
*.zip binary

# LFS (uncomment as needed)
# *.pth filter=lfs diff=lfs merge=lfs -text

# Exclude from GitHub Linguist language detection
*.p linguist-detectable=false
*.gz linguist-detectable=false
```

## .gitignore

Ignore OS artifacts, caches, build outputs, and local data directories.

```gitignore
# OS
thumbs.db
.DS_Store

# Internal references, dependencies, temporary folders & files
.env
**/__ref__/
*.log
*.lock
*.db
*.db.*
*.sqlite3*
*.sqlite3.*
*.parquet
.cache/

# Agents
tasks/

# Docker
/docker_data/

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info
.pytest_cache
.ipynb_checkpoints
venv/

# pip
**/build/

# pytest-cov
**/coverage.xml
**/.coverage*
/junit
/htmlcov

# ruff
.ruff_cache/

# jest-cov
**/coverage/*

# JavaScript
**/node_modules/
**/logs
**/dist/

# db related
**/ch_data/
**/vm_data/
**/ch_logs/
/db/
**/vl_data/
**/rqlite_db/
```

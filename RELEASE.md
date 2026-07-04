# Release process

This document explains how `polars-fuzzy` ships prebuilt wheels to PyPI so
that users can run `pip install polars-fuzzy` with **no Rust toolchain
required**.

## How it works

`.github/workflows/release.yml` triggers on a `v*` tag. It builds wheels for
every supported platform × Python combo, plus an sdist, and publishes them to
PyPI using **Trusted Publishing (OIDC)** — no API token stored as a secret.

| Platform           | Targets                                  |
|--------------------|------------------------------------------|
| Linux              | `x86_64`, `aarch64` (manylinux 2_28)     |
| Windows            | `x86_64`                                 |
| macOS              | `x86_64-apple-darwin`, `aarch64-apple-darwin` |
| Python             | cp39, cp310, cp311, cp312, cp313         |

Combinations without a matching wheel fall back to the sdist (source build,
**does** require Rust).

---

## One-time setup: register the Trusted Publisher

Do this once on PyPI. You must own the project name `polars-fuzzy` on PyPI
(first publish from an account, or reserve via the published-file flow).

1. Go to **https://pypi.org/manage/account/publishing/**.
2. Under **Add a new publisher → GitHub**, fill in:
   - **PyPI Project Name:** `polars-fuzzy`
   - **Owner:** `Pratham-26`
   - **Repository:** `rust_helpers`
   - **Workflow name:** `release.yml`
   - **Environment name:** `release`
3. Save. PyPI shows a "pending publisher" until the first run lands.

> Tip: repeat on **https://test.pypi.org** to smoke-test publishing before
> your first real release. See "Smoke test" below.

---

## Cutting a release

```bash
# 1. Bump version in BOTH files (they must match).
#    Cargo.toml   -> version = "0.2.0"
#    pyproject.toml -> version = "0.2.0"
$EDITOR Cargo.toml pyproject.toml

git add Cargo.toml pyproject.toml
git commit -m "chore: release v0.2.0"

# 2. Tag and push.
git tag v0.2.0
git push origin main --tags
```

Pushing the tag kicks off `release.yml`. Watch it under the **Actions** tab.
When the `release` job finishes:

- The wheels + sdist are live on PyPI.
- A GitHub Release is created with the wheels attached.

Verify:

```bash
python -m venv /tmp/check && source /tmp/check/bin/activate
pip install polars-fuzzy
python -c "import polars_fuzzy as pf; print(pf.__name__)"
```

---

## Smoke test against TestPyPI (optional, recommended before first real release)

1. Register a Trusted Publisher on **test.pypi.org** exactly as above.
2. Edit `.github/workflows/release.yml`, find the `Publish to PyPI` step,
   and uncomment the `repository-url` line:
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
3. Push a throwaway tag (e.g. `v0.0.0-rc1`).
4. Install from TestPyPI and sanity-check:
   ```bash
   pip install -i https://test.pypi.org/simple/ polars-fuzzy
   ```
5. Revert the `repository-url` change before the real release.

---

## Troubleshooting

- **`pending publisher` never resolves:** the workflow filename, environment
  name (`release`), or repo owner doesn't match what you entered on PyPI.
- **`unable to find interpreter`:** maturin needs Python present on the
  runner; the `setup-python` step covers this. Don't remove it.
- **aarch64-linux build fails:** the `before-script-linux` step installs
  `gcc-aarch64-linux-gnu`; if the image changes, confirm that package still
  exists.
- **Wheels missing a Python version:** maturin-action builds for every
  interpreter the `setup-python`/container provides. cp313 is set as the
  host interpreter; older ABI tags are produced automatically via
  `--interpreter` discovery inside the action.

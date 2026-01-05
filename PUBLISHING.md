# Publishing ShipML to PyPI

This guide explains how to publish ShipML to PyPI so users can install it with `pip install shipml`.

## Prerequisites

1. **PyPI Account**
   - Create account at https://pypi.org/account/register/
   - Verify your email
   - Enable 2FA (required for publishing)

2. **TestPyPI Account** (for testing)
   - Create account at https://test.pypi.org/account/register/
   - Verify email

3. **API Tokens**
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - Save tokens securely (you'll only see them once!)

## Step 1: Update Version

Before each release, update the version in `pyproject.toml`:

```toml
[project]
version = "0.1.0"  # Change this for new releases
```

Commit the version change:
```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.1.0"
git tag v0.1.0
git push origin main --tags
```

## Step 2: Build the Package

Clean previous builds and build fresh:

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
.venv/bin/python -m build
```

This creates:
- `dist/shipml-X.X.X.tar.gz` (source distribution)
- `dist/shipml-X.X.X-py3-none-any.whl` (wheel)

## Step 3: Test on TestPyPI First

**IMPORTANT:** Always test on TestPyPI before publishing to real PyPI!

```bash
# Upload to TestPyPI
.venv/bin/twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your TestPyPI API token>
```

Test installation:
```bash
# Create fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple shipml

# Test it works
shipml --version
shipml serve examples/sklearn_example.py  # If you have a test model

# Clean up
deactivate
rm -rf test_env
```

## Step 4: Publish to PyPI

Once testing passes, publish to real PyPI:

```bash
.venv/bin/twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your PyPI API token>
```

## Step 5: Verify Installation

```bash
# In a fresh environment
pip install shipml

# Test
shipml --version
```

## Step 6: Create GitHub Release

1. Go to https://github.com/prabhueshwarla/shipml/releases/new
2. Choose the tag you created (v0.1.0)
3. Title: "ShipML v0.1.0"
4. Description: Copy from CHANGELOG.md (see below)
5. Attach the dist files (optional)
6. Publish release

## Using .pypirc (Optional)

To avoid entering credentials every time:

```bash
# Create ~/.pypirc
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-token>
EOF

# Secure the file
chmod 600 ~/.pypirc
```

Then upload without prompts:
```bash
.venv/bin/twine upload --repository testpypi dist/*
.venv/bin/twine upload dist/*
```

## Release Checklist

- [ ] All tests pass (`.venv/bin/pytest tests/`)
- [ ] Pre-push checks pass (`./pre_push.sh`)
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated with release notes
- [ ] Committed and pushed to GitHub
- [ ] Git tag created (`git tag v0.1.0`)
- [ ] Package built (`python -m build`)
- [ ] Tested on TestPyPI
- [ ] Published to PyPI
- [ ] Tested installation (`pip install shipml`)
- [ ] GitHub release created
- [ ] Announcement posted (Twitter, Reddit, HN)

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New feature (custom pipelines)
- `0.9.0` → `1.0.0`: Stable release, breaking changes

## Troubleshooting

### "Invalid or non-existent authentication information"
- Regenerate API token on PyPI
- Make sure you're using `__token__` as username
- Token should start with `pypi-`

### "File already exists"
- You can't re-upload the same version
- Bump version number and rebuild
- Delete dist/ and rebuild

### "Package name already taken"
- Someone else owns the name on PyPI
- Choose a different name in `pyproject.toml`
- Check availability: https://pypi.org/project/shipml/

### "README rendering failed"
- Check README with: `.venv/bin/twine check dist/*`
- Fix any RST/Markdown issues
- Rebuild and test again

## Automation (Future)

Consider setting up GitHub Actions to automate:
- Build on every tag push
- Upload to PyPI automatically
- Create GitHub release

Example workflow: `.github/workflows/publish.yml`

## References

- [PyPI Publishing Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging User Guide](https://packaging.python.org/)

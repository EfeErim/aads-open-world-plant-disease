# Public smoke samples

These images are programmatically generated geometric plant-like inputs for I/O and notebook smoke tests.

- They contain no personal data and were not scraped from the web.
- They are covered by the repository MIT license.
- They are **not** training data and **not** accuracy evidence.

This boundary is intentional: the original research datasets are excluded because their file-level redistribution
licenses are not sufficiently documented for a public repository.

Checksums and roles are recorded in [`manifest.json`](manifest.json). Regenerate everything with:

```bash
python -m pip install pillow
python scripts/generate_public_assets.py
```

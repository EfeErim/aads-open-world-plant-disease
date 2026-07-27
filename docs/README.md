# Documentation

The root [README](../README.md) is the shortest entry point.

- [Architecture overview](architecture/overview.md)
- [Code organization map](architecture/code_organization_map.md)
- [Public result records](../evidence/)
- [Dataset layout](../data/README.md)
- [Training workflow](../src/workflows/training.py)
- [Inference workflow](../src/workflows/inference.py)

Generated reports, private dataset manifests and internal project-management notes are intentionally not copied into
the public repository. The maintained implementation is under `src/`; notebooks and scripts call that code instead
of carrying separate model implementations.

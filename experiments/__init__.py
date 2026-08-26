"""Repository-local experiment package.

The explicit package marker is a provenance and import-boundary control: it
prevents a later regular ``experiments`` package on a caller-controlled
``PYTHONPATH`` from replacing this repository's experiment modules.
"""

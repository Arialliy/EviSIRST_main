"""EviSIRST public model package.

The frozen V3 implementation keeps its original module names because those
names are part of the checkpoint architecture contract.  The source files are
stored under ``model/_internal`` and added to this package's search path so
existing frozen imports continue to resolve without cluttering the public
directory.
"""

from pathlib import Path


_INTERNAL_PATH = str(Path(__file__).resolve().parent / "_internal")
if _INTERNAL_PATH not in __path__:
    __path__.append(_INTERNAL_PATH)


__all__ = []


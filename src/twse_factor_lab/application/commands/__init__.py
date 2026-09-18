"""Thin, importable command entry points.

Command modules own argument parsing and dependency assembly. Domain packages
remain independent of this package.
"""

__all__ = []
"""Application command owners.

Migrated commands expose ``main(argv)`` and a testable workflow. Root runners
remain compatibility or frozen-replay entry points until their cohort clears
the inventory gates.
"""

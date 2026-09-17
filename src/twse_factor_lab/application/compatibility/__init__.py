"""Compatibility boundaries for legacy and frozen replay entry points.

Adapters remain intentionally small and one-way. They may import application
owners, but reusable domain packages must never depend on this namespace.
"""

__all__ = []

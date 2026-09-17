"""Compatibility adapter for the application research-report command."""

from twse_factor_lab.application.commands.research_report import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())

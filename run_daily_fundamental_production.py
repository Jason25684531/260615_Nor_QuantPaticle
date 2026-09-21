"""Compatibility adapter for the daily Fundamental command."""

from twse_factor_lab.application.commands.fundamental_final import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())

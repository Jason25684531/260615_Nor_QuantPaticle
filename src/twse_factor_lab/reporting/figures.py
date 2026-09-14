"""The seven controlled, headless research-report figures."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FigureResult:
    figure_id: str
    status: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        path: str | None = None
        if self.path is not None:
            path = (
                self.path.resolve().relative_to(root.resolve()).as_posix()
                if root is not None
                else str(self.path)
            )
        return {
            "figure_id": self.figure_id,
            "status": self.status,
            "path": path,
            "metadata": self.metadata,
        }


def _series(values: Any, value_column: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.copy()
    elif isinstance(values, pd.DataFrame):
        frame = values.copy()
        if "date" in frame:
            frame = frame.set_index("date")
        column = value_column or ("returns" if "returns" in frame else frame.columns[0])
        result = frame[column]
    elif isinstance(values, list) and values and isinstance(values[0], dict):
        frame = pd.DataFrame(values)
        result = frame.set_index("date")[value_column or "value"]
    else:
        result = pd.Series(values)
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result = pd.to_numeric(result, errors="coerce").sort_index()
    return result[~result.index.duplicated(keep="first")]


def _available(values: Any, value_column: str | None = None) -> pd.Series | None:
    try:
        result = _series(values, value_column)
    except (KeyError, TypeError, ValueError):
        return None
    return result if not result.dropna().empty else None


def _save(
    figure_id: str,
    output_dir: str | Path,
    draw: Any,
    metadata: dict[str, Any],
) -> FigureResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{figure_id}.png"
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(fig, ax)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return FigureResult(figure_id, "GENERATED", path, metadata)
    except Exception as exc:
        if path.exists():
            path.unlink()
        return FigureResult(
            figure_id,
            "UNAVAILABLE",
            None,
            {**metadata, "error": f"{type(exc).__name__}: {exc}"},
        )
    finally:
        if fig is not None:
            plt.close(fig)


def _unavailable(
    figure_id: str,
    metadata: dict[str, Any],
    reason: str,
    output_dir: str | Path | None = None,
) -> FigureResult:
    if output_dir is not None:
        stale = Path(output_dir) / f"{figure_id}.png"
        if stale.exists():
            stale.unlink()
    return FigureResult(
        figure_id,
        "UNAVAILABLE",
        None,
        {**metadata, "reason": reason},
    )


def build_fig_01_cumulative_returns(
    returns: Any,
    output_dir: str | Path,
    benchmark: Any = None,
) -> FigureResult:
    series = _available(returns, "returns")
    metadata = {
        "source": "strategy_handoff/returns.csv",
        "nan_handling": "NaN filled with 0.0 before compounding",
        "benchmark": "AVAILABLE" if benchmark is not None else "UNAVAILABLE",
    }
    if series is None:
        return _unavailable(
            "01_cumulative_returns", metadata, "returns are empty", output_dir
        )

    def draw(_fig: Any, ax: Any) -> None:
        curve = (1.0 + series.fillna(0.0)).cumprod() - 1.0
        ax.plot(curve.index, curve, label="Strategy")
        if benchmark is not None:
            benchmark_series = _available(benchmark)
            if benchmark_series is not None:
                ax.plot(
                    (1.0 + benchmark_series.fillna(0.0)).cumprod() - 1.0,
                    label="Benchmark",
                )
            else:
                metadata["benchmark"] = "UNAVAILABLE"
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Cumulative return")
        ax.legend(loc="best")

    return _save("01_cumulative_returns", output_dir, draw, metadata)


def _pyfolio_figure(
    figure_id: str,
    returns: Any,
    output_dir: str | Path,
    function_name: str,
    **kwargs: Any,
) -> FigureResult:
    series = _available(returns, "returns")
    metadata = {
        "source": "strategy_handoff/returns.csv",
        "engine": "pyfolio-reloaded",
        "api": f"pyfolio.plotting.{function_name}",
        "nan_handling": "drop NaN observations; all-NaN unavailable",
    }
    if series is None:
        return _unavailable(figure_id, metadata, "returns are empty", output_dir)
    try:
        from pyfolio import plotting

        plotter = getattr(plotting, function_name)
    except Exception as exc:
        return _unavailable(
            figure_id,
            metadata,
            f"pyfolio API unavailable: {type(exc).__name__}: {exc}",
            output_dir,
        )

    def draw(_fig: Any, ax: Any) -> None:
        plotter(series.dropna(), ax=ax, **kwargs)

    result = _save(figure_id, output_dir, draw, metadata)
    return result


def build_fig_02_underwater_drawdown(
    returns: Any, output_dir: str | Path
) -> FigureResult:
    return _pyfolio_figure(
        "02_underwater_drawdown", returns, output_dir, "plot_drawdown_underwater"
    )


def build_fig_03_rolling_sharpe(
    returns: Any, output_dir: str | Path, rolling_window: int = 126
) -> FigureResult:
    result = _pyfolio_figure(
        "03_rolling_sharpe",
        returns,
        output_dir,
        "plot_rolling_sharpe",
        rolling_window=rolling_window,
    )
    return FigureResult(
        result.figure_id,
        result.status,
        result.path,
        {**result.metadata, "rolling_window": rolling_window},
    )


def build_fig_04_rolling_volatility(
    returns: Any,
    output_dir: str | Path,
    rolling_window: int = 126,
    annualization_sessions: int = 252,
) -> FigureResult:
    series = _available(returns, "returns")
    metadata = {
        "source": "strategy_handoff/returns.csv",
        "engine": "report-native",
        "rolling_window": rolling_window,
        "annualization_sessions": annualization_sessions,
        "nan_handling": "drop NaN observations; all-NaN unavailable",
    }
    if series is None:
        return _unavailable(
            "04_rolling_volatility", metadata, "returns are empty", output_dir
        )

    def draw(_fig: Any, ax: Any) -> None:
        volatility = series.dropna().rolling(rolling_window).std(ddof=0) * np.sqrt(
            annualization_sessions
        )
        ax.plot(volatility.index, volatility)
        ax.set_title("Annualized Rolling Volatility")
        ax.set_ylabel("Volatility")

    return _save("04_rolling_volatility", output_dir, draw, metadata)


def build_fig_05_monthly_returns(returns: Any, output_dir: str | Path) -> FigureResult:
    return _pyfolio_figure(
        "05_monthly_returns",
        returns,
        output_dir,
        "plot_monthly_returns_heatmap",
    )


def build_fig_06_portfolio_exposure(
    positions: Any,
    equity: Any,
    output_dir: str | Path,
    prices: Any = None,
) -> FigureResult:
    metadata = {
        "source": "strategy_handoff/positions.csv + nav.csv",
        "engine": "report-native",
        "exposure_definition": "sum(abs(actual dollar positions excluding cash)) / actual equity",
        "positions_contract": "dollar_positions_including_cash",
        "nan_handling": "drop rows with unavailable equity",
    }
    if not isinstance(positions, pd.DataFrame):
        try:
            positions = pd.DataFrame(positions)
        except (TypeError, ValueError):
            return _unavailable(
                "06_portfolio_exposure", metadata, "positions unavailable", output_dir
            )
    frame = positions.copy()
    if "date" in frame:
        frame = frame.set_index("date")
    try:
        frame.index = pd.to_datetime(frame.index)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        equity_series = _series(equity, "nav").reindex(frame.index)
    except (KeyError, TypeError, ValueError):
        return _unavailable(
            "06_portfolio_exposure",
            metadata,
            "positions or equity unavailable",
            output_dir,
        )
    asset_columns = [column for column in frame if str(column).lower() != "cash"]
    if not asset_columns or equity_series.dropna().empty:
        return _unavailable(
            "06_portfolio_exposure",
            metadata,
            "positions or equity are empty",
            output_dir,
        )
    actual = frame[asset_columns]
    if prices is not None:
        price_frame = (
            prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
        )
        if "date" in price_frame:
            price_frame = price_frame.set_index("date")
        price_frame.index = pd.to_datetime(price_frame.index)
        price_frame = price_frame.reindex(frame.index).apply(
            pd.to_numeric, errors="coerce"
        )
        shared = [column for column in asset_columns if column in price_frame]
        if shared:
            actual = actual[shared] * price_frame[shared]
            metadata["price_input_used"] = True
    else:
        metadata["price_input_used"] = False
    valid = equity_series.notna() & equity_series.ne(0)
    exposure = actual.abs().sum(axis=1).div(equity_series.abs()).where(valid).dropna()
    if exposure.empty:
        return _unavailable(
            "06_portfolio_exposure", metadata, "no valid equity rows", output_dir
        )
    long_only = bool((actual.dropna(how="all") >= 0).all().all())
    metadata["long_only"] = long_only
    if long_only:
        metadata["net_equals_gross"] = True

    def draw(_fig: Any, ax: Any) -> None:
        ax.plot(exposure.index, exposure)
        ax.set_title("Realized Portfolio Exposure")
        ax.set_ylabel("Gross exposure / equity")

    return _save("06_portfolio_exposure", output_dir, draw, metadata)


def build_fig_07_is_vs_oos_nav(
    is_nav: Any,
    oos_nav: Any,
    output_dir: str | Path,
    oos_boundary: str | None = None,
) -> FigureResult:
    is_series = _available(is_nav, "nav")
    oos_series = _available(oos_nav)
    metadata = {
        "source": "IS handoff nav.csv + fresh_state_oos.json.oos_nav",
        "engine": "report-native",
        "oos_execution_mode": "FRESH-STATE RE-EXECUTION",
        "oos_evidence_type": "fresh_state_reexecution",
        "oos_boundary": oos_boundary,
        "nan_handling": "drop NaN observations; each NAV series normalized to its first value",
    }
    if is_series is None or oos_series is None:
        return _unavailable(
            "07_is_vs_oos_nav",
            metadata,
            "IS or fresh-state OOS NAV unavailable",
            output_dir,
        )

    def draw(_fig: Any, ax: Any) -> None:
        is_normalized = is_series / is_series.iloc[0]
        oos_normalized = oos_series / oos_series.iloc[0]
        ax.plot(is_normalized.index, is_normalized, label="IS")
        ax.plot(oos_normalized.index, oos_normalized, label="OOS")
        if oos_boundary:
            boundary = pd.Timestamp(oos_boundary)
            ax.axvline(boundary, color="black", linestyle="--", label="OOS boundary")
        ax.set_title("IS vs Fresh-state OOS NAV")
        ax.set_ylabel("Normalized NAV")
        ax.legend(loc="best")

    return _save("07_is_vs_oos_nav", output_dir, draw, metadata)


def _read_series(path: Path | None, column: str) -> pd.Series | None:
    if path is None or not path.exists():
        return None
    try:
        return _available(pd.read_csv(path), column)
    except (OSError, ValueError, KeyError):
        return None


def build_figures(model: Any, output_dir: str | Path) -> list[FigureResult]:
    """Build exactly the seven figure slots from model source-file references."""

    output_dir = Path(output_dir)
    source = model.performance.data.get("source_files", {})
    root = Path(source.get("repository_root", "."))
    returns = _read_series(
        root / source["returns"] if source.get("returns") else None, "returns"
    )
    nav = _read_series(root / source["nav"] if source.get("nav") else None, "nav")
    positions_path = root / source["positions"] if source.get("positions") else None
    positions = None
    if positions_path is not None and positions_path.exists():
        try:
            positions = pd.read_csv(positions_path)
        except (OSError, ValueError):
            positions = None
    oos_data = model.oos.data
    oos_nav = None
    oos_source = oos_data.get("source_file")
    if oos_source:
        try:
            payload = json_load(root / oos_source)
            oos_nav = payload.get("oos_nav")
        except (OSError, ValueError, TypeError):
            oos_nav = None
    annualization = model.pyfolio.data.get("metadata", {}).get(
        "annualization_sessions", 252
    )
    results = [
        build_fig_01_cumulative_returns(returns, output_dir),
        build_fig_02_underwater_drawdown(returns, output_dir),
        build_fig_03_rolling_sharpe(returns, output_dir),
        build_fig_04_rolling_volatility(
            returns, output_dir, annualization_sessions=int(annualization)
        ),
        build_fig_05_monthly_returns(returns, output_dir),
        build_fig_06_portfolio_exposure(positions, nav, output_dir),
        build_fig_07_is_vs_oos_nav(
            nav,
            oos_nav,
            output_dir,
            oos_data.get("oos_start"),
        ),
    ]
    return results


def json_load(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "FigureResult",
    "build_fig_01_cumulative_returns",
    "build_fig_02_underwater_drawdown",
    "build_fig_03_rolling_sharpe",
    "build_fig_04_rolling_volatility",
    "build_fig_05_monthly_returns",
    "build_fig_06_portfolio_exposure",
    "build_fig_07_is_vs_oos_nav",
    "build_figures",
]

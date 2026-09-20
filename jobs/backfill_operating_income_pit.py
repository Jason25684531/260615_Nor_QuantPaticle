"""Backfill canonical operating-income PIT rows from cached MOPS income reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from twse_factor_lab.data.fundamental import RawFundamentalCache, parse_mops_statement


def _record_id(row: pd.Series) -> str:
    fields = ("ticker", "metric", "period_end", "publication_date", "value")
    return hashlib.sha256("|".join(map(str, row[list(fields)])).encode()).hexdigest()


def _raw_operating_income(cache: Path) -> pd.DataFrame:
    url = "https://mopsov.twse.com.tw/mops/web/ajax_t163sb04"
    rows = []
    for year in range(102, 115):
        for season in range(1, 5):
            payload = {
                "encodeURIComponent": "1",
                "step": "1",
                "firstin": "1",
                "off": "1",
                "year": year,
                "season": season,
                "TYPEK": "sii",
            }
            key = RawFundamentalCache._key(url, {"params": {}, "data": payload})
            path = cache / f"{key}.raw"
            if not path.exists():
                raise RuntimeError(f"MISSING_MOPS_INCOME_CACHE:{year}Q{season}")
            parsed = parse_mops_statement(
                path.read_text(encoding="utf-8"), year=year, season=season
            )
            rows.append(
                parsed.loc[parsed["metric"].eq("operating_income")].dropna(
                    subset=["value"]
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    keys = ["ticker", "roc_year", "season"]
    ambiguous = frame.groupby(keys)["value"].nunique().gt(1)
    if ambiguous.any():
        raise RuntimeError("AMBIGUOUS_OPERATING_INCOME_SOURCE")
    return frame.drop_duplicates(keys, keep="last")


def build(root: Path, *, write: bool = False) -> dict[str, int]:
    path = root / "data/processed/fundamental_pit_v2/fundamental_records.parquet"
    records = pd.read_parquet(path)
    eps = records.loc[records["metric"].eq("eps")].copy()
    eps["roc_year"] = pd.to_datetime(eps["period_end"]).dt.year - 1911
    eps["season"] = pd.to_datetime(eps["period_end"]).dt.quarter
    raw = _raw_operating_income(root / "data/raw/fundamental")
    merged = eps.merge(
        raw[["ticker", "roc_year", "season", "value"]],
        on=["ticker", "roc_year", "season"],
        how="left",
        suffixes=("", "_operating"),
    )
    matched = merged["value_operating"].notna()
    operating = eps.iloc[matched.to_numpy()].drop(columns=["roc_year", "season"]).copy()
    operating["metric"] = "operating_income"
    operating["value"] = merged.loc[matched, "value_operating"].to_numpy()
    operating["source"] = "mops+doc.twse"
    operating["source_components"] = "mops_income_statement"
    operating["formula_version"] = pd.NA
    operating["metric_semantics"] = "cumulative-as-reported"
    operating["source_record_id"] = operating.apply(_record_id, axis=1)
    operating["revision_status"] = "ORIGINAL"
    result = pd.concat(
        [records.loc[~records["metric"].eq("operating_income")], operating],
        ignore_index=True,
    ).sort_values(["ticker", "metric", "period_end", "publication_date"])
    result = result.reset_index(drop=True)
    summary = {
        "eps_records": len(eps),
        "operating_income_records": len(operating),
        "source_missing_records": int((~matched).sum()),
        "source_missing_tickers": sorted(
            merged.loc[~matched, "ticker"].astype(str).unique().tolist()
        ),
        "output_records": len(result),
    }
    if write:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".parquet", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            result.to_parquet(temporary, index=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        (path.parent / "operating_income_backfill_audit.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), write=args.write), sort_keys=True))

# Data Lineage

```text
MOPS / TWSE fundamental data
  -> PIT available_date / period_end
  -> Canonical Fundamental Store
  -> Canonical Runtime Provider
  -> G2 Operating Income YoY + G3 EPS YoY
  -> 50/50 score -> rank -> Top5 -> score weighted -> REB60 target
  -> recommendation -> persistence -> Web / LINE
```

Research, Shadow, and Production invoke separate adapters over the same shared
canonical factor, score, ranking, selection, target, and REB60 logic.

# dashboard/data

Small aggregated CSVs for the Power BI dashboard, built from the cleaned pipeline output:

```bash
python -m src.clean                    # full pipeline, writes these CSVs at the end
python -m src.clean --dashboard-only   # rebuild only these CSVs from data/processed/*.parquet
```

**Status: not generated yet (TBD, run `python -m src.clean`).** This repository was built in
fallback mode without the DfT data. No CSV is committed until it comes from the real data,
so synthetic numbers can never be mistaken for real ones. `src/clean.py` refuses to write any
file over 5 MB.

## Files

| File | Grain | Key columns | Notes |
|---|---|---|---|
| `monthly_collisions_by_severity.csv` | month | `month` (first day of month) | collisions, fatal / serious / slight collisions, KSI, adjusted serious and adjusted KSI collisions |
| `yearly_totals.csv` | year | `year`, `year_start` | collision counts by severity, plus casualty counts: casualties, killed, serious, slight, adjusted serious / slight, KSI casualties |
| `by_speed_limit.csv` | year × speed limit | `year`, `year_start`, `category` | same collision measures as the monthly file |
| `by_light_conditions.csv` | year × light condition | `year`, `year_start`, `category` | " |
| `by_road_type.csv` | year × road type | `year`, `year_start`, `category` | " |
| `by_region.csv` | year × region | `year`, `year_start`, `category` | region is mapped from police force (approximate) |
| `fatal_collision_points.csv` | one fatal collision | `collision_index`, `date`, `latitude`, `longitude` | only rows with valid coordinates (`coord_valid`) |

A missing category is written as `Missing` so it stays visible in totals. "Adjusted" columns
are sums of DfT's per-record probabilities, so they are not whole numbers.

## Date table spec

Created in Power BI with DAX (full code in `../measures.md`), not loaded from a file:

| Column | Type | Example | Purpose |
|---|---|---|---|
| `Date` | date | 2016-03-01 | key; one row per day, 2005-01-01 to 2019-12-31, no gaps |
| `Year` | whole number | 2016 | slicers, axis |
| `Month Number` | whole number | 3 | sort key for `Month` |
| `Month` | text | Mar | axis label (sort by `Month Number`) |
| `Year Month` | text | 2016-03 | monthly axis label |
| `Quarter` | text | Q1 | optional grouping |

Relationships (single direction, many-to-one, Date table on the "one" side):
- `monthly_collisions_by_severity[month]` → `Date[Date]`
- `yearly_totals[year_start]` → `Date[Date]`
- each `by_*[year_start]` → `Date[Date]`
- `fatal_collision_points[date]` → `Date[Date]`

Mark `Date` as the date table (Table tools → Mark as date table → `Date[Date]`) so
`SAMEPERIODLASTYEAR` works.

Licence: Contains public sector information licensed under the Open Government Licence v3.0.
Source: Department for Transport, Road Safety Data (STATS19).

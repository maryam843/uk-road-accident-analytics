# Power BI: DAX measures and build guide

Tables come from `dashboard/data/*.csv` (see `dashboard/data/README.md`). Every measure below
is written against those column names. The `.pbix` file is not committed (gitignored).

## 1. Date table

```DAX
Date =
ADDCOLUMNS (
    CALENDAR ( DATE ( 2005, 1, 1 ), DATE ( 2019, 12, 31 ) ),
    "Year", YEAR ( [Date] ),
    "Month Number", MONTH ( [Date] ),
    "Month", FORMAT ( [Date], "mmm" ),
    "Year Month", FORMAT ( [Date], "yyyy-mm" ),
    "Quarter", "Q" & QUARTER ( [Date] )
)
```

Then: mark as date table on `Date[Date]`; sort `Month` by `Month Number`; create the
relationships listed in `dashboard/data/README.md`.

## 2. Core measures (collision table)

```DAX
Collisions =
SUM ( monthly_collisions_by_severity[collisions] )

Fatal Collisions =
SUM ( monthly_collisions_by_severity[fatal_collisions] )

Serious Collisions =
SUM ( monthly_collisions_by_severity[serious_collisions] )

Slight Collisions =
SUM ( monthly_collisions_by_severity[slight_collisions] )

KSI Collisions =
SUM ( monthly_collisions_by_severity[ksi_collisions] )

KSI Collisions (adjusted) =
SUM ( monthly_collisions_by_severity[ksi_collisions_adjusted] )

KSI Share (adjusted) =
DIVIDE ( [KSI Collisions (adjusted)], [Collisions] )
```

## 3. Year-on-year (time intelligence)

```DAX
Collisions PY =
CALCULATE ( [Collisions], SAMEPERIODLASTYEAR ( 'Date'[Date] ) )

Collisions YoY % =
DIVIDE ( [Collisions] - [Collisions PY], [Collisions PY] )
```

Why this form: `SAMEPERIODLASTYEAR` shifts the *current filter context* back exactly one
year, so in a 2016 cell `Collisions PY` is the 2015 total and nothing more. A filter like
`'Date'[Year] < MAX('Date'[Year])` would sum **all** earlier years. Against one year's total,
that produces a large, false year-on-year fall. `DIVIDE` returns BLANK for 2005 (no prior year)
instead of an error or infinity. Format `Collisions YoY %` as a percentage.

## 4. People killed is a different measure from fatal collisions

```DAX
-- Collision table: number of collisions in which at least one person died.
Fatal Collisions (yearly) =
SUM ( yearly_totals[fatal_collisions] )

-- Casualty table: number of people who died. One fatal collision can kill several people.
People Killed =
SUM ( yearly_totals[killed] )

Serious Casualties =
SUM ( yearly_totals[serious_casualties] )

Serious Casualties (adjusted) =
SUM ( yearly_totals[serious_casualties_adjusted] )

Slight Casualties =
SUM ( yearly_totals[slight_casualties] )

All Casualties =
SUM ( yearly_totals[casualties] )
```

Never label `Fatal Collisions` as "deaths". `People Killed ≥ Fatal Collisions` in every year;
the card tooltips should say which table each number comes from.

## 5. Reconciliation check (data-quality card)

```DAX
Severity Reconciliation Check =
[Collisions] - ( [Fatal Collisions] + [Serious Collisions] + [Slight Collisions] )
```

Must show **0** for any filter selection. A non-zero value means a severity code was lost
or double-counted. `tests/test_clean.py::test_dashboard_data` asserts the same identity on
the CSVs. Add a second card for casualties:

```DAX
Casualty Reconciliation Check =
SUM ( yearly_totals[casualties] )
    - ( [People Killed] + [Serious Casualties] + [Slight Casualties] )
```

## 6. Breakdown measures (speed limit, light, road type, region)

```DAX
Collisions (by speed limit) =
SUM ( by_speed_limit[collisions] )

KSI Share adjusted (by speed limit) =
DIVIDE ( SUM ( by_speed_limit[ksi_collisions_adjusted] ), SUM ( by_speed_limit[collisions] ) )
```

Repeat with `by_light_conditions`, `by_road_type` and `by_region` (same column names).

## 7. Build guide

| Page | Visual | Fields / measures | Notes |
|---|---|---|---|
| Overview | 4 KPI cards | `Collisions`, `Collisions YoY %`, `Fatal Collisions`, `People Killed` | each card subtitle names its table (collision vs casualty) |
| Overview | Line chart | X = `Date[Year Month]`, Y = `Collisions` | monthly trend; one axis only |
| Overview | Line chart | X = `Date[Year]`, Y = `KSI Collisions` and `KSI Collisions (adjusted)` | footnote below (required) |
| Conditions | 4 bar charts | Y = `category`; X = `KSI Share adjusted (by …)` | sort descending; tooltip adds collision count |
| Map | Map / Azure Map | Latitude = `fatal_collision_points[latitude]`, Longitude = `…[longitude]` | for a scatter-style map: **X = longitude, Y = latitude** |
| Data quality | 2 cards | `Severity Reconciliation Check`, `Casualty Reconciliation Check` | should read 0 |

**Filter scope**
- A `Date[Year]` slicer on each page, synced across pages (View → Sync slicers). It filters
  every table through the `Date` relationships.
- Page-level filter on the Conditions page only: `category` is not `Missing` (state this in a
  text box). Do not apply it report-wide, or the totals stop matching the Overview page.
- No visual-level filters on the KPI cards, so they always show the full slicer selection.

**Footnote for adjusted KSI (paste under every chart that uses an adjusted measure):**

> KSI = killed or seriously injured. Adjusted figures are adjusted for changes in police
> severity reporting (DfT): from 2016 many forces moved to injury-based reporting (CRASH/COPA),
> which records more injuries as serious. Contains public sector information licensed under
> the Open Government Licence v3.0. Source: Department for Transport, Road Safety Data (STATS19).

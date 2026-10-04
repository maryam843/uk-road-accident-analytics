# Data licence

The road safety data used in this project is **not** stored in this repository.
It is downloaded at run time from the Department for Transport (DfT).

Contains public sector information licensed under the Open Government Licence v3.0. Source: Department for Transport, Road Safety Data (STATS19).

- Licence: Open Government Licence v3.0 —
  https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- Dataset: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Official totals (table RAS0101):
  https://www.gov.uk/government/statistical-data-sets/reported-road-accidents-vehicles-and-casualties-tables-for-great-britain

Road traffic counts (notebook 08): annual average daily flow (AADF) by count point.
Contains public sector information licensed under the Open Government Licence v3.0. Source:
Department for Transport, Road traffic statistics.

- Licence: Open Government Licence v3.0 (as above)
- Downloads page: https://roadtraffic.dft.gov.uk/downloads (file `dft_traffic_counts_aadf.zip`
  and the metadata `all-traffic-data-metadata.pdf`, both linked from that page)

Files under `tests/fixtures/data_guide_codes.csv` are a subset of the code list in the
DfT Road Safety Open Dataset data guide (OGL v3.0, same attribution as above).
All other fixture rows are synthetic and do not describe real collisions or real traffic counts.

The source code in this repository is licensed separately under the MIT licence (see `LICENSE`).

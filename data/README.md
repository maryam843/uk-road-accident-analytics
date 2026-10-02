# data/

Nothing in this folder is committed except this README. `raw/`, `processed/` and `kaggle/`
are gitignored.

| Folder | Contents | Created by |
|---|---|---|
| `raw/` | DfT collision and casualty CSVs (1979-latest), the data guide `.xlsx`, `ras0101.ods`, `DOWNLOAD_LOG.json` | `python -m src.download` |
| `processed/` | `collisions_2005_2019.parquet`, `casualties_2005_2019.parquet`, monthly and yearly aggregates, `CLEAN_LOG.json` | `python -m src.clean` |
| `kaggle/` | Optional: `ukroadaccident.csv`, supplied by hand, used only by notebook 00 to document why it was rejected | you |

Set `DATA_DIR` to put this folder somewhere else (for example on Google Drive in Colab):

```bash
export DATA_DIR=/path/to/data
```

Licence: see `../DATA_LICENSE.md`.

# Data Setup

## Included in Git (demo / dashboard)

| File | Purpose |
|------|---------|
| `modis_data_sample.csv` | Subset of NASA MODIS active-fire detections (India) |

## Full dataset (local only)

| File | Purpose |
|------|---------|
| `modis_data10%.csv` | 10% stratified sample of full MODIS export (app default when present) |
| `modis_data.csv` | Full MODIS active-fire dataset |

**How to obtain:** Download from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) or your archived export; place in `data/`.

**Resolution order:** `app/app.py` and the notebook try `modis_data10%.csv`, then `modis_data_sample.csv`, then `modis_data.csv`.

**Run dashboard:** `streamlit run app/app.py`

# Predictability collapse in urban evolution

This repository contains the reproducible code used for a TCULU-based analysis of urban redevelopment predictability in Chinese cities. It includes the city extraction workflow, per-city feature-construction pipeline, model evaluation scripts, permutation-importance analysis, robustness checks and figure-generation utilities.

The repository is intended to accompany the manuscript *Predictability Collapse in Urban Evolution*. Large raster inputs, OpenStreetMap extracts and derived city-period result tables are not stored in this code repository. Those data products should be archived separately as Supplementary Data / Zenodo Dataset records.

## Repository Structure

```text
src/
  auto_city_preprocess/   City extraction, administrative splitting and OSM road clipping
  pipeline/               Per-city TCULU pipeline, feature construction and model analysis
scripts/
  batch/                  Batch deployment and 79-city pipeline runners
  analysis/               National summary tables, audits and robustness analyses
  figures/                Manuscript and extended-data figure generation
  submission/             Supplementary workbook and package-checking utilities
docs/
  DATA_REQUIREMENTS.md    Required external inputs and generated outputs
  UPLOAD_CHECKLIST.md     GitHub / Zenodo release checklist
```

## Software Environment

The pipeline was run with Python 3.11 on macOS. Core dependencies are listed in `requirements.txt`.

Create an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some supplementary workbook utilities use Node.js scripts for spreadsheet packaging. These are optional and only needed when regenerating the final submission workbook.

## Main Workflow

1. Extract candidate city windows from national TCULU rasters.
2. Split or constrain candidate built-up components by administrative boundaries.
3. Clip OSM roads by administrative boundary and connect them to each city pipeline.
4. Construct cell, neighbourhood, street-network syntax and functional-morphology features.
5. Define dynamic built-up extents from the previous period and classify built-to-built redevelopment.
6. Train period-specific classifiers and run ablation, permutation-importance and PDP analyses.
7. Aggregate 79-city outputs into manuscript tables, supplementary data and figures.

## Minimal Per-City Run

After preparing a city folder with `data/`, `figs/` and `pipeline/`, run:

```bash
cd path/to/city/tculu_pipeline_v9_final
python pipeline/run_all.py
```

The dynamic redevelopment scope is built in `09.5_dynamic_builtup_extent.py`. The final modelling steps are:

```bash
python pipeline/10_assemble_dataset.py
python pipeline/11_train_and_ablate.py
python pipeline/12_shap_analysis.py
python pipeline/13_update_transition_stats.py
python pipeline/14_period_shap_analysis.py
```

Note: `12_shap_analysis.py` is named after the historical pipeline file, but the feature attribution used in the manuscript is permutation importance, not exact SHAP values.

## Data Availability

External and derived data are documented in `docs/DATA_REQUIREMENTS.md`. The code assumes that TCULU rasters, Geofabrik OSM shapefiles and derived 79-city outputs are available locally or downloaded from the associated data archive.

## License

Code is released under the MIT License unless otherwise noted.


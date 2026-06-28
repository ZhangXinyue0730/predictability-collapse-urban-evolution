# Reproducibility Notes

The analysis combines geospatial preprocessing, dynamic built-up masking, supervised classification and post-hoc model diagnostics. Because raw TCULU and OSM data are large, reproducibility is organized in two layers:

1. **Code layer**: this GitHub repository, archived as software.
2. **Data layer**: Supplementary Data / Zenodo Dataset, containing the final city-period tables needed to reproduce manuscript figures and statistical tests without rerunning all raster pipelines.

For full raw-data reproduction, users need to download TCULU rasters and OSM extracts, then run the city extraction and per-city pipeline scripts. For manuscript-result reproduction, users can run the figure and analysis scripts against the archived Supplementary Data workbook.


# Upload Checklist

## GitHub Code Repository

1. Create or open the GitHub repository.
2. Upload this package as repository contents.
3. Confirm that no raw TCULU rasters, OSM shapefiles or local-only result folders are included.
4. Create a release, for example `v1.0.0`.
5. Link the GitHub repository to Zenodo and archive the release.
6. Fill the manuscript Code availability statement with the GitHub URL and Zenodo software DOI.

## Zenodo Dataset Record

1. Upload the Supplementary Data workbook and any required CSV result matrices.
2. Set Resource type to `Dataset`.
3. Use a title such as `Data from "Predictability Collapse in Urban Evolution"`.
4. Add related works and grant information if available.
5. Publish the record and copy the DOI into the Data availability statement.

## Before Release

Run these checks:

```bash
rg "PRIVATE_LOCAL_PATH_PATTERN" .
rg "\\.tif|\\.npy|\\.npz|\\.pkl|\\.shp|\\.gpkg" .
```

The first command should return no private local paths in files intended for public release. The second command should only match documentation or `.gitignore`, not actual large files.

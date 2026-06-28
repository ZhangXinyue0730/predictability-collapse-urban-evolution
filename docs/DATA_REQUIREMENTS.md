# Data Requirements

This code repository does not include raw or derived large data files. The following inputs are required to reproduce the analysis.

## Raw Inputs

1. **TCULU urban land-use rasters** for 1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020 and 2024.
2. **OpenStreetMap road extracts** downloaded from Geofabrik, typically provincial-level `*-latest-free.shp.zip` archives.
3. **Administrative boundary shapefiles** used to split continuous built-up regions and clip OSM roads.
4. **City statistical yearbook tables** used for population, GDP, per-capita GDP and sectoral GDP covariates.

## Core Derived Outputs

The manuscript-level Supplementary Data workbook should contain:

1. `SD1_Master_79x8`: 79-city by 8-period master table.
2. `SD2_Multimodel_AUC_AP`: AUC/AP matrices for gradient boosting, random forest and L2 logistic regression.
3. `SD3_Ablation_M0_M7`: M0-M7 ablation results.
4. `SD4_Permutation_Importance`: full feature-level and family-level permutation importance.
5. `SD5_Transitions_All`: complete land-use transition matrix, not only top transitions.
6. `SD6_Update_Expansion_Area`: redevelopment and expansion area summaries.
7. `SD7_Regional_Tests`: regional and city-group statistical tests.
8. `SD8_Syntax_Summary`: street-network and syntax summaries.
9. `SD9_Feature_Dictionary_167`: feature dictionary for the full 167-dimensional feature set.
10. `SD10_Source_Index`: mapping between manuscript figures, tables and source scripts.

## Land-Use Classes

The built-to-built redevelopment definition uses previous-period dynamic built-up extents and excludes vacant / unused land from redevelopment transitions. The working land-use code table should be recorded in the Supplementary Information and Supplementary Data.

## Archiving Recommendation

Archive the Supplementary Data workbook and any non-public derived CSV files as a Zenodo Dataset. Archive this GitHub repository as a Zenodo Software record after creating a GitHub release.


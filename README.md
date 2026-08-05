# README

## Overview

This repository contains the COSIPY repository and scripts used within the manuscript titled "Towards the Bayesian calibration of a glacier surface energy balance model for unmonitored glaciers". 

## Repository Contents

- `SCRIPTS/` – This folder holds the various scripts used to analyse and generate output or intermediary files of COSIPY or the forcing
- `cosipy/utilities/` – This subfolder holds the internal utilities to generate among others the static file or the [HORAYZON](https://github.com/ChristianSteger/HORAYZON) fields.
- `cali_data` – Holds the glacier-mean albedo and snowline altitude calibration data. In addition, we added the .csv file outputs of the LHS sampling (first and second stage)

## Requirements
Running the scripts compiled below requires various python packages. First and foremost, an installation of all [COSIPY](https://cositools-cosipy.readthedocs.io/en/latest/) requirements is necessary. Second, the analysis involves many additional python packages including tensorflow, pymc, arviz and many more. Therefore, we do not provide a singular environment file but instead hope the user can generate multiple environments according to their needs. In case the exact python and package version is desired, we provide them upon request.

### Forcing Data
To be able to run this setup, we downloaded COSMO-CLM simulations from the [Earth System Grid Federation](https://esgf-ui.ceda.ac.uk/search).

### Calibration Data
We derived the calibration data by either directly downloading it or adjusting it from the scripts provided below. We performed manual checks for both the snowline and albedo data. The calibration data can be found in the folder `cali_data`. 
Calibration data and the required code can be found at:

- Hugonnet et al. (2021): https://doi.org/10.1038/s41586-021-03436-z
- Ren et al. (2024): https://doi.org/10.1016/j.oneear.2024.08.010
- Loibl et al. (2025): https://doi.org/10.1038/s41597-024-04309-6

## Repository Structure

The analyses and data preparation scripts are grouped by their leading letter:

| Prefix | Description |
|--------|-------------|
| **A** | Forcing Data Preparation |
| **B** | Calibration workflow |
| **C** | Analysis and plotting |
| **R** | Additions to address the reviewers comments |

### A – Forcing Data Preparation

| File | Description |
|------|-------------|
| `A_01_CORDEX-DKRZ_preprocess_fixandprepareprec.ipynb` | This script chaotically checks the precipitation data downloaded from ESGF and creates a new snowfall field using the precipitation phase partitioning method.|
| `A_02_CORDEX-DKRZ_lapse-rate-sensitivty.ipynb` | Computes lapse rates (T2M, RH, snowfall, precipitation) from COSMO model fields and compares it to the AWS observations by fitting linear regressions over small grid boxes around the HEF. |
| `A_03_CORDEX-DKRZ_prepare-COSIPY-from-COSMO_utils.py` | This utility module prepares COSMO fields for COSIPY forcing: it loads/subsets/reprojects datasets against a shapefile, fixes units/timestamps, computes derived fields (U2, interpolated pressure, snowfall in m), and assembles a COSIPY-ready dataframe/CSV. |

### B – Calibration workflow

| File | Description |
|------|-------------|
| `B_01_create-local-sens-experiments.py` | Generates a table of parameter combinations for a manual sensitivity study (albedos, roughness lengths, precip scaling, aging, etc.), sampling each parameter over linear ranges. These must be parsed into COSIPY (for example using spotpy_run_fromlist.py).|
| `B_02_Landsat-Albedo-preparation.ipynb` | Reads in the individual filtered Landsat albedo scenes and merges them into the calibration dataset. |
| `B_03_COSIPY-param-sensitivity-in-prior.ipynb` | Generates the figures and analyses of the local sensitivity runs. |
| `B_04_Wide-LHS-screening.ipynb` | Evaluates the output of the first 500 LHS samples (generated from `LHS_surrogate_parameters.py`) and filters it by the likelihood using a 3-sigma filtering rule to serve as input for the second LHS bounds. To continue, the boudns need to be parsed back into LHS_surrogate_parameters.py. |
| `B_05_Narrow-LHS-ClusterPCA.ipynb` | Filters the narrow LHS ensemble (COSIPY output from `LHS_surrogate_parameters.py`) by likelihood using the 3-sigma filtering rule, runs PCA and clustering to identify well‑performing parameter regions, and derives constrained priors (truncated normals) for MCMC. |
| `B_06_create_NN_emulator.ipynb` | Builds and trains a neural‑network emulator to predict COSIPY outputs from LHS model parameters, evaluates performance via train/validation and k‑fold cross‑validation, plots diagnostics, and saves the trained model and CV results. |
| `B_07_Misc_create_NN_Nsamples_analysis.ipynb` | Tests neural‑network emulator sensitivity to training sample size. Performs repeated subsampling, trains models for different sample counts, evaluates MAE/RMSE/R² for mass balance, snowline and albedo. |
| `B_08_Eval-NN-nsamples.ipynb` | Analyses the output of the previous script, creating a summarising figure showing the metric distributions and median trends versus sample size. |
| `B_09_eval-point-LHS.ipynb` | Evaluates point-scale COSIPY LHS ensemble output (generated using `point_LHS_surrogate_parameters.py`) against AWS observations. Filters the best-performing ensemble members using a joint RMSE-based threshold and builds weighted ensemble. |

### C – Analysis and plotting

| File | Description |
|------|-------------|
| `C_01_Evaluate_CSPY_at_AWS.ipynb` | Loads COSIPY output and AWS observations for the same period, compares them across key variables, and creates evaluation figures showing COSIPY vs AWS performance. |
| `C_02_COSIPY_evaluate_hypsometry.ipynb` | Loads COSIPY hypsometry ensemble outputs and WGMS/AWS reference data, compares simulated mass-balance vs observations across elevation bands and creates a summary figure. |
| `C_02_COSIPY_evaluate_hypsometry.ipynb` | Loads COSIPY hypsometry ensemble outputs and WGMS/AWS reference data, compares simulated mass-balance vs observations across elevation bands and creates a summary figure. |
| `C_03_explore-pymc-results.ipynb` | Analyzes emulator-based MCMC posterior results for COSIPY calibration, including trace plots, parameter posterior summaries, convergence diagnostics, and predictive checks. Run twice by switching commented data, first to extract initial values for the follow-up MCMC run, then for the final evaluation. |
| `C_04_point-SEB-comparison.ipynb` | Compares point-scale COSIPY and AWS surface energy balance and meteorological variables using weighted ensemble runs, then produces time-series, distribution, and scatter plots for fluxes and state variables. |


### R – Review comments

| File | Description |
|------|-------------|
| `R_01_check-bounds-impact.py` | Evaluates how pushing the albedo of fresh snow and ice beyond their prior bounds changes the log-likelihood of the geodetic mass balance, snowline altitude, and albedo observations. Requires manual runs of COSIPY with the step-wise changes to the albedo parameters. |
| `R_02_check-MB-fit.py` | Compares glaciological mass-balance records (WGMS and Klug et al.) with the decade-long geodetic Hugonnet estimate, and then inspects the relationship between normalized snowline altitude and observed albedo. |
| `R_03_analyze_postmed-addedbias.py` | Loads posterior-run daily outputs and the same MCMC observations, then evaluates this single posterior median with fixed forcing biases against geodetic MB, albedo, and snowline observations. Requires an existing COSIPY simulation ran with the fixed biases. |

In addition, many scripts exist that were necessary to run the many COSIPY simulations and summary statistics and generate the analysis presented in this manuscript. These are:

- emulator_firststage_mcmc.py: first stage of the MCMC runs using the constructed emulator
- emulator_secondstage_mcmc.py: second stage of the MCMC runs using the priors derived from the first stage
- LHS_surrogate_parameters.py: spotpy based implementation of the LHS runs
- point_LHS_surrogate_parameters.py: spotpy based implementation of the point scale LHS runs
- postprocess_chainsmerge.py: processing utility to concat the various individual arviz .nc files into one posterior file
- preprocess_initvals.py: generates the initial starting values for the MCMC chains using LHS
- spotpy_run_fromlist.py: takes a .csv with parameter values and parses these into COSIPY to run cosipy from a list, used e.g., for the local sensitivity runs
- spotpy_finalmcmc_ensemble.py: does the same as above, but uses the final 300 sample members from the MCMC
- create_eb_at_aws.py: takes COSIPY output at closest grid cell in elevation to the AWS data and stores it to a .csv
- create_eb_subsets.py: takes the weighted glacier-means of the COSIPY outputs and stores a subset of variables
- create_hypsometry.py: groups COSIPY data into WGMS 50m elevation bins to calculate annual metrics for specific variables
- create_LHS_albedodata.py: calculates area-weighted spatial averages for albedo mass balance and melt output including grouping it into seasonal and annual values. Needs to be run for each subset list of COSIPY simulations (e.g., local sensitivity runs, or LHS runs)
- create_LHS_AWSmetrics.py: compares COSIPY output against daily AWS values at target elevation and stores daily comparison time series and metrics across all ensemble runs
- create_LHS_EB_lookup.py: processes COSIPY simulations to compute area-weighted energy balance metrics split into seasonal totals
- create_LHS_snowline_ts.py: archived utility (not up to date) to generate full snowline time series for each COSIPY simulation in the ensemble
- prepare_aws_comparison_hef.py: extracts COSIPY variables for grid cells corresponding to the two AWS elevations, normalises surface heights and aggregates data across all ensemble runs into variable-specific csv files for both station locations
- xr_aggregate_monthly.py: computes glacier-wide spatial averages for energy balance and mass balance variables and saves monthly mean climatologies. Also compiles the mass balance time series across all ensemble members into a single csv table

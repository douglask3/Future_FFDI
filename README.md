# FFDI Analysis and Plotting Tools

This repository contains various Python and R scripts used for analyzing and visualizing fire weather data, specifically focusing on the Forest Fire Danger Index (FFDI) and its relationships with meteorological drivers. The repository includes code for calculating and plotting FFDI thresholds, time between FFDI categories, consensus plots, and percentiles for regional analysis. 

## Repository Structure

### Files

- **`ffdi_variables_analysis.py`**  
  Code for background processing, preparing the data for plotting and fitting relationships between FFDI and its drivers.

- **`Variables_Analysis3.ipynb`**  
  Jupyter Notebook for plotting and fitting relationships between FFDI and its drivers.

- **`calc_consensus_plots.py`**  
  Creates consensus plots, adapted from Chris Kent's consensus code. Generates consensus plots based on the processed FFDI data.

- **`calc_threshold_exceedance.py`**  
  Calculates the number of times the FFDI exceeds a specified daily threshold for each 20-year period of interest (threshold exceedance).

- **`centile_calc_table_cp.py`**  
  Reads in anomalies of the number of days above the FFDI threshold, averages for a region, and calculates the 10th, 50th, and 90th percentiles. Outputs these values in a table for each FFDI category.

- **`ffdi_time_between_vals_v2.r`**  
  R script that plots the amount of time in each FFDI category for each region, using the Python script `calc_threshold_exceedance.py`.

- **`mass_comands.sh`**  
  Shell script that grabs data off Mass and runs the relevant initial commands for executing the rest of the analysis. If you already have the data downloaded, you just need to run the following two commands separately:

  ```bash
  python calc_threshold_exceedance.py
  python calc_consensus_plots.py
  ```

  
## Usage

### Requirements
A few dependencies are required:

- Python 3.x
- R (for `ffdi_time_between_vals_v2.r`)- **pandas**: Data manipulation and analysis.
- **cftime**: Support for handling dates and times in netCDF files.
- **iris**: A library for working with climate and weather data, including multidimensional arrays.
- **numpy**: Fundamental package for scientific computing with Python.
- **matplotlib**: Plotting library for creating static, animated, and interactive visualizations in Python.
- **cartopy**: A library for cartographic projections and geospatial visualizations.
- **ascend**: (Please ensure you have the correct version based on your system setup. This package is used for specialized computations in the project.)
- **glob**: Used for file path manipulation and pattern matching.
- **iris.coord_categorisation**: Used for categorizing coordinates in Iris cubes.
- **iris.plot**: Used for plotting data from Iris cubes.
- **iris.quickplot**: A simplified plotting interface for Iris.
- **matplotlib.colors**: A module from Matplotlib for working with colors in visualizations.
- **numpy.ma**: A module from NumPy for working with masked arrays.
- **pdb**: Python debugger (used for debugging, typically in development stages).

For some, you may also need the constrain_cubes_standard fucntion that can be found at https://github.com/douglask3/Bayesian_fire_models/blob/main/libs/constrain_cubes_standard.py

### Setup and Execution

#### Running the Python Scripts
To run the primary Python analysis scripts, ensure you have all required libraries installed (such as NumPy, pandas, Matplotlib, and any other dependencies). Once your environment is set up:

1. **Process the data for threshold exceedance**.

2. **Generate consensus plots**.

#### Running the R Script
The R script (`ffdi_time_between_vals_v2.r`) can be used to visualize the time spent in each FFDI category. Make sure you have R installed and the necessary libraries available.

Run the R script by executing:
  ```bash
  Rscript ffdi_time_between_vals_v2.r
  ```


#### Running the Shell Script (`mass_comands.sh`)
If you have your data already, simply run the following command to grab the necessary data and run the initial scripts for processing:


```bash
bash mass_comands.sh
```


This script will automatically download data and execute the relevant analysis steps (running `calc_threshold_exceedance.py` and `calc_consensus_plots.py`).

## Output

The scripts will generate various outputs, including:

- Consensus plots
- Tables of threshold exceedance
- Percentile calculations for FFDI anomalies


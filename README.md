# Engineered Biological Neural Networks as Basic Logic Operators – Official Code Repository

This repository contains the complete codebase accompanying the publication:

"Engineered biological neural networks as basic logic operators"  
Authors: Küchler, J., Vulić, K., et al.  
Published in: Frontiers in Computational Neuroscience, 2025  
[DOI](https://doi.org/10.1101/2024.12.23.630065)  
[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.23.630065v2)

The repository provides all scripts and tools necessary to reproduce the results and figures presented in the paper. It includes data processing routines, spike analysis modules, experiment scripts, and plotting utilities for CMOS-based neural recordings.

If you encounter any issues or have questions regarding the code or its usage, feel free to reach out (contact info below).

-------------------------------------------------------------------------------

## Installation (Linux + Conda)

We recommend using Conda for environment and dependency management. The setup has been tested with Python 3.10.

1. Create and activate a new environment:
```bash
conda create --name cmos_env python=3.10
conda activate cmos_env
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Note: Some optional parts of the pipeline depend on the IDTxl toolbox for information-theoretic analysis. Due to dependency constraints, this is not included in the requirements file. Please refer to the IDTxl documentation for manual installation:  
https://github.com/pwollstadt/IDTxl

## External Dependencies

This repository uses Python packages provided with the MaxWell Biosystems recording system. These are not included here due to licensing restrictions.

You must have access to the proprietary Python API provided with your licensed system.

Please contact MaxWell Biosystems or refer to your installation materials for setup instructions.


-------------------------------------------------------------------------------

## Data Availability

The processed data used in this publication can be found [here](https://doi.org/10.3929/ethz-b-000729744). 

-------------------------------------------------------------------------------

## Code Structure

/data/SignalProcessing/
- analysisAlgorithms.py: Contains calculations used for capacity estimation.
- dataPreparation.py: Used for restructuring data efficiently.
- evaluate.py: Functions to extract metrics to quantify responses to a stimulus.
- mapping.py: Color codings used to visualize electrodes on a microstructure.
- postprocessing.py: Contains spike detection methods, blanking and filtering to extract spike times.

/data/utils/
- cmos_utils.py: CMOS-specific low-level utilities.
- parse_utils.py: Configuration and argument parsing.
- h5pyUtils.py: .h5 data loading and handling.

/data/plotting.py  
Reusable plotting utilities (e.g., colormaps).

/interface/
- selectElectrodes.py: Interface for electrode selection from the microelectrode array.  
  Note: This file requires a manual edit in line 243 to adapt to your system.

/experiments/inputSweep/
Scripts used for processing the main input sweep experiments from the manuscript.

- biohybridProcess2D.py: Spike extraction and artefact blanking.
- biohybrid2DAdvancedProcessing.py: Spike shape clustering, capacity computation, response analysis.
- biohybridPlot2D.py: Plotting and PDF generation.
- config.py: Contains all experiment-specific parameters and paths.

Equivalent scripts exist for 1D experiments.

-------------------------------------------------------------------------------

## Notebooks

- 1DExperiment_Plotting.ipynb  
- Capacity.ipynb  

These notebooks are used for visualizations.

-------------------------------------------------------------------------------

## Notes

- This repository assumes access to specific .h5 files containing recorded data. These are not included.
- Code has been tested on Linux with Python 3.10.

-------------------------------------------------------------------------------

## LICENSE

This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
See the license file for details or visit [Creative Commons](https://creativecommons.org/licenses/by-nc/4.0/).

-------------------------------------------------------------------------------

## Citation

If you use this code in your work, please cite:

@article{kuchler2024engineered,  
  title     = {Engineered biological neural networks as basic logic operators},  
  author    = {K{\"u}chler, Jo{\"e}l and Vuli{\'c}, Katarina and Yao, Haotian and Valmaggia, Christian and Ihle, Stephan J and Weaver, Sean and V{\"o}r{\"o}s, J{\'a}nos},  
  journal   = {bioRxiv},  
  pages     = {2024--12},  
  year      = {2024},  
  publisher = {Cold Spring Harbor Laboratory}  
}

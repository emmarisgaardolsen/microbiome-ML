# Predictive Modelling of Health  Status based on Human  Microbiome Data  🧬 🦠

<p align="center">
  Emma Risgaard Olsen (eol@post.au.dk) <br>
  <em>Data Science, Prediction, and Forecasting (F24)</em>
  <br>
  Aarhus University, Cognitive Science MSc.
  <br>
</p>
<hrZ>


## Project description
Repository for performing machine learning analyses on human microbiome data .
 
## Repository Structure 
The structure below is to be updated.
```
/microbiome-ML
|-- LICENCSE
|-- data/                              # Dataset directory, data not included due to size issues but is outputted from code contained in the repo
|-- src/                                 # Code for replicating full analysis pipeline
    |-- Python               # Python code for data pre-processing, modelling and visualisation
    |-- R                           # R code for getting the data
|-- requirements.txt        # Project dependencies
|-- figs/                                 # Figures and result tables 
|-- README.md                # The top-level guide for using this project
|-- .gitignore                      # Specifies intentionally untracked files to ignore
|-- setup.sh                        # Setup script 
```

## Installation
The current project was developed to work using Visual Studio Code version: 1.89.1 (Universal) on a machine running MacOS Sonoma 14.1 on a M1 Pro chip. Any unix-based system should be able to run the code, but the installation instructions may vary.  

Follow the below steps to run the code as intended: 
1. Clone the current repository
```
git clone https://github.com/emmarisgaardolsen/microbiome-ML.git
cd microbiome-ML
```

2. Run `setup.sh`
To replicate the results, you first need to setup the environment. Run the `setup.sh` script in your bash terminal. The script automatically creates a virtual environment for the project, activates the virtual environment and installs the necessary dependencies/packages with the correct versions. 

```
bash setup.sh
```

## Usage

The extracted data is not included in this GitHub repo as its size exceeds the limits of GitHub storage. However, the data can be outputted from running the code in the `get_data.Rmd` in the `src/R/` directory and subsequently preprocessing it through the `backup_preproc.ipynb` code in `src/Python/’. 


0. `/work/EmmaRisgaardOlsen#9993/microbiome-ML/src/R/get_data.Rmd` for fetching data (data is already fetched and stored in the `data/` directory)
1. `EDA.ipynb` for some exploratory data analysis 
2. `preproc.ipynb` for preprocessing the data (creating outcome variable and making data splits)


## Contact 
For any inquiries regarding the project or collaboration, please contact Emma Risgaard Olsen at eol@post.au.dk.

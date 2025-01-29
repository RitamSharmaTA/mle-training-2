

# Median Housing Value Prediction

This project predicts median housing values using machine learning models. The dataset can be downloaded from [here](https://raw.githubusercontent.com/ageron/handson-ml/master/). A script is provided to automatically fetch the data.

## Models Used

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

## Project Workflow

1. **Data Preparation and Cleaning**: The data is cleaned, with missing values handled via imputation.
2. **Feature Engineering**: New features are created, and correlations between variables are analyzed.
3. **Sampling and Splitting**: The dataset is split into training and test sets using various sampling techniques.
4. **Model Training and Evaluation**: The models (Linear Regression, Decision Tree, and Random Forest) are trained and evaluated using Root Mean Squared Error (RMSE) as the performance metric.

## Installation and Usage

### 1. Environment Setup

Ensure you have `conda` installed, then create and activate the environment:

```bash
# Create the environment from the environment file
conda env create -f env.yml

# Activate the environment
conda activate mle-dev
```

### 2. Install the Package

Install the code package in editable mode:

```bash
pip install -e .
```

### 3. Execute the Workflow

You can now run the following commands to execute different steps of the process:

#### Ingest Data:

```bash
ingest_data --output-path data
```

This command downloads and extracts the housing dataset to the specified output path.

#### Train a Model:

```bash
train --input-path data/housing.csv --output-path models/linearmodel.pkl --model-type linear
```

This command trains a model of your choice. You can choose from:

- `"linear"` for Linear Regression
- `"tree"` for Decision Tree Regressor
- `"forest"` for Random Forest Regressor

The trained model will be saved at the specified output path.

#### Score the Model:

```bash
score --model-path models/linearmodel.pkl --data-path data/housing.csv --output-path results/lrscore.txt
```

This command evaluates the trained model on the test data and outputs the Root Mean Squared Error (RMSE) to the specified file.

## Sphinx Documentation

Sphinx documentation has been added for detailed usage and API references. To generate the documentation, follow these steps:

1. Navigate to the `docs` directory.
2. Run the following command:

```bash
make html
```

This will generate the HTML documentation, which can be accessed in the `_build/html` folder.


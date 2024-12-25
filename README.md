

# Median Housing Value Prediction

This project predicts the median housing value based on a given housing dataset. The dataset can be downloaded from [here](https://raw.githubusercontent.com/ageron/handson-ml/master/). The script includes functionality to automatically download the data.

The following machine learning techniques have been used to build the predictive model:

- **Linear Regression**
- **Decision Tree**
- **Random Forest**

## Steps Performed

1. **Data Preparation and Cleaning**: We prepare and clean the data by checking and imputing missing values.
2. **Feature Engineering**: Relevant features are generated, and variables are analyzed for correlations.
3. **Sampling Techniques**: Multiple sampling techniques are evaluated, and the dataset is split into training and test sets.
4. **Model Training and Evaluation**: The aforementioned models (Linear Regression, Decision Tree, and Random Forest) are trained and evaluated using Mean Squared Error (MSE) as the final evaluation metric.

## To Execute the Script

### 1. Environment Setup

Ensure that you have `conda` installed on your system. Then, follow these steps to set up the environment:

```bash
# Create the environment from the environment file
conda env create -f env.yml

# Activate the newly created environment
conda activate mle-dev
```

### 2. Run the Script

Once the environment is activated, run the script to execute the housing value prediction:

```bash
python nonstandardcode.py
```

This will run the code and generate the results based on the data and models described.


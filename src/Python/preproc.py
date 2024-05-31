import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle as pkl
from sklearn import datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
import skbio
from skbio.stats.composition import clr
from sklearn.model_selection import train_test_split
from skbio.stats.composition import closure, clr
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# ---- Partition data 
selected_variables = [
    'age', 'gender', 'country', 'diet', 'smoker', 'ever_smoker', 'alcohol', 'diet'
]

# drop all columns that are not 'healthy' or in the selected_variables list
data = data[[col for col in data.columns if 'healthy' in col or col in selected_variables]]

# One-hot encode categorical variables with 0/1 encoding
categorical_vars = ['gender', 'country', 'diet', 'smoker', 'ever_smoker','alcohol']
demographic_data = pd.get_dummies(data, columns=categorical_vars, drop_first=True, dtype=int)

# Update selected_variables after one-hot encoding
encoded_variables = list(demographic_data.columns)
selected_variables = [var for var in encoded_variables if any(orig_var in var for orig_var in ['age', 'gender', 'country', 'diet', 'smoker', 'alcohol'])]

# Drop rows with any NaN values
data = demographic_data.dropna()

# Scale age column
scaler = StandardScaler()
data['age'] = scaler.fit_transform(data[['age']])

# Split data into features (X) and target (y)
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42,stratify=y_train)

# Save the data
datasets = {
    'Training set': (X_train, y_train),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / 'baseline'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')

    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data = combined_data.dropna(how='all')  # Ensure no empty rows before saving
    combined_data.to_csv(output_dir / csv_name, index=False)


# %% [markdown]
# # Baseline undersampled

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# ---- Partition data 
selected_variables = [
    'age', 'gender', 'country', 'diet', 'smoker', 'ever_smoker', 'alcohol', 'diet'
]

# drop all columns that are not 'healthy' or in the selected_variables list
data = data[[col for col in data.columns if 'healthy' in col or col in selected_variables]]

# One-hot encode categorical variables with 0/1 encoding
categorical_vars = ['gender', 'country', 'diet', 'smoker', 'ever_smoker','alcohol']
demographic_data = pd.get_dummies(data, columns=categorical_vars, drop_first=True, dtype=int)

# Update selected_variables after one-hot encoding
encoded_variables = list(demographic_data.columns)
selected_variables = [var for var in encoded_variables if any(orig_var in var for orig_var in ['age', 'gender', 'country', 'diet', 'smoker', 'alcohol'])]

# Drop rows with any NaN values
data = demographic_data.dropna()

# Scale age column
scaler = StandardScaler()
data['age'] = scaler.fit_transform(data[['age']])

# split data into features (X) and target (y)
X = data[selected_variables]
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Identify minority class and count
minority_class = y_train.value_counts().idxmin()
minority_count = y_train.value_counts().min()

# Separate majority and minority classes
X_train_minority = X_train[y_train == minority_class]
y_train_minority = y_train[y_train == minority_class]

X_train_majority = X_train[y_train != minority_class]
y_train_majority = y_train[y_train != minority_class]

# Undersample majority class
X_train_majority_downsampled = X_train_majority.sample(n=minority_count, random_state=42)
y_train_majority_downsampled = y_train_majority.sample(n=minority_count, random_state=42)

# Combine the downsampled majority class with the minority class
X_train_balanced = pd.concat([X_train_minority, X_train_majority_downsampled])
y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])

# Save the data
datasets = {
    'Training set': (X_train_balanced, y_train_balanced),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / 'baseline_undersampled'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)


# %% [markdown]
# # Baseline SMOTE

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# ---- Partition data 
selected_variables = [
    'age', 'gender', 'country', 'diet', 'smoker', 'ever_smoker', 'alcohol', 'diet'
]

# drop all columns that are not 'healthy' or in the selected_variables list
data = data[[col for col in data.columns if 'healthy' in col or col in selected_variables]]
# One-hot encode categorical variables with 0/1 encoding

categorical_vars = ['gender', 'country', 'diet', 'smoker', 'ever_smoker','alcohol']
demographic_data = pd.get_dummies(data, columns=categorical_vars, drop_first=True, dtype=int)

# Update selected_variables after one-hot encoding
encoded_variables = list(demographic_data.columns)
selected_variables = [var for var in encoded_variables if any(orig_var in var for orig_var in ['age', 'gender', 'country', 'diet', 'smoker', 'alcohol'])]

# Drop rows with any NaN values
data = demographic_data.dropna()

# Scale age column
scaler = StandardScaler()
data['age'] = scaler.fit_transform(data[['age']])

# split data into features (X) and target (y)
X = data[selected_variables]
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.15, 
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=X_test.shape[0] / X_train.shape[0], random_state=42,stratify=y_train)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# Save the data
datasets = {
    'Training set': (X_train_smote, y_train_smote),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / 'baseline_smote'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# # 50species

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter away meta data by defining all bacteria columns as containing the string "|" or the outcome variable
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# For each entry in the bacteria columns, remove all strings that come before the last 's__' but keep 's__', also keeping healthy column
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Extract each line in the 50_species.txt file and put it in a list
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Define matching columns
matching_columns = [col for col in data.columns if col in fifty_species or col == 'healthy']
data = data[matching_columns]

# Split data into features (X) and target (y)
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42,stratify=y_train)

# Save the data
datasets = {
    'Training set': (X_train, y_train),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / '50SPC'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')

    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data = combined_data.dropna(how='all')  # Ensure no empty rows before saving
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# 

# %% [markdown]
# # 50species undersampled

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter bacteria columns
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# Modify bacteria columns
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Load fifty species
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Filter data to include matching columns
matching_columns = [col for col in data.columns if col in fifty_species or col == 'healthy']
data = data[matching_columns]

# Split data into features and target
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42,stratify=y_train)

# Identify minority class and count
minority_class = y_train.value_counts().idxmin()
minority_count = y_train.value_counts().min()

# Separate majority and minority classes
X_train_minority = X_train[y_train == minority_class]
y_train_minority = y_train[y_train == minority_class]

X_train_majority = X_train[y_train != minority_class]
y_train_majority = y_train[y_train != minority_class]

# Undersample majority class
X_train_majority_downsampled = X_train_majority.sample(n=minority_count, random_state=42)
y_train_majority_downsampled = y_train_majority.sample(n=minority_count, random_state=42)

# Combine the downsampled majority class with the minority class
X_train_balanced = pd.concat([X_train_minority, X_train_majority_downsampled])
y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])

# Save the data
datasets = {
    'Training set': (X_train_balanced, y_train_balanced),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / '50SPC_undersampled'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# # 50species SMOTE

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter away meta data by defining all bacteria columns as containing the string "|" or the outcome variable
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# For each entry in the bacteria columns, remove all strings that come before the last 's__' but keep 's__', also keeping healthy column
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Extract each line in the 50_species.txt file and put it in a list
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Define matching columns
matching_columns = [col for col in data.columns if col in fifty_species or col == 'healthy']
data = data[matching_columns]

# Split data into features (X) and target (y)
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42,stratify=y_train)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Save the data
datasets = {
    'Training set': (X_train_smote, y_train_smote),
    'Validation set': (X_val, y_val),
    'Test set': (X_test, y_test)
}

output_dir = root / 'data' / '50SPC_smote'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# # 50species CLR

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

# Remove entries with missing 'study_condition'
data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter away meta data by defining all bacteria columns as containing the string "|" or the outcome variable
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# For each entry in the bacteria columns, remove all strings that come before the last 's__' but keep 's__', also keeping healthy column
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Split data into features (X) and target (y)
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

# Apply CLR transformation
X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert the CLR transformed data back to DataFrame
X_train_clr = pd.DataFrame(X_train_clr, index=X_train.index, columns=X_train.columns)
X_val_clr = pd.DataFrame(X_val_clr, index=X_val.index, columns=X_val.columns)
X_test_clr = pd.DataFrame(X_test_clr, index=X_test.index, columns=X_test.columns)

# Extract each line in the 50_species.txt file and put it in a list
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Define matching columns
matching_columns = [col for col in X_train_clr.columns if col in fifty_species]

# Filter data to keep only the matching columns
X_train_clr = X_train_clr[matching_columns]
X_val_clr = X_val_clr[matching_columns]
X_test_clr = X_test_clr[matching_columns]

# Recreate datasets with target column
datasets = {
    'Training set': (X_train_clr, y_train),
    'Validation set': (X_val_clr, y_val),
    'Test set': (X_test_clr, y_test)
}

# Save the data
output_dir = root / 'data' / '50SPC_CLR'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')

    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data = combined_data.dropna(how='all')  # Ensure no empty rows before saving
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# # 50species CLR undersampled

# %%
from skbio.stats.composition import clr

# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter bacteria columns
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# Modify bacteria columns
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Load fifty species
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Filter data to include matching columns
matching_columns = [col for col in data.columns if col in fifty_species or col == 'healthy']
data = data[matching_columns]

# Split data into features and target
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

# Apply CLR transformation
X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert the CLR transformed data back to DataFrame
X_train_clr = pd.DataFrame(X_train_clr, index=X_train.index, columns=X_train.columns)
X_val_clr = pd.DataFrame(X_val_clr, index=X_val.index, columns=X_val.columns)
X_test_clr = pd.DataFrame(X_test_clr, index=X_test.index, columns=X_test.columns)

# Identify minority class and count
minority_class = y_train.value_counts().idxmin()
minority_count = y_train.value_counts().min()

# Separate majority and minority classes
X_train_minority = X_train_clr[y_train == minority_class]
y_train_minority = y_train[y_train == minority_class]

X_train_majority = X_train_clr[y_train != minority_class]
y_train_majority = y_train[y_train != minority_class]

# Undersample majority class
X_train_majority_downsampled = X_train_majority.sample(n=minority_count, random_state=42)
y_train_majority_downsampled = y_train_majority.sample(n=minority_count, random_state=42)

# Combine the downsampled majority class with the minority class
X_train_balanced = pd.concat([X_train_minority, X_train_majority_downsampled])
y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])

# Save the data
datasets = {
    'Training set': (X_train_balanced, y_train_balanced),
    'Validation set': (X_val_clr, y_val),
    'Test set': (X_test_clr, y_test)
}

output_dir = root / 'data' / '50SPC_CLR_undersampled'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)


# %% [markdown]
# # 50species CLR SMOTE

# %%
# Load data
root = Path.cwd().parents[1]
path = root / "data" / "raw" / "full_df_with_meta.csv"
data = pd.read_csv(path)

data = data.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
data = data[data['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
data['healthy'] = np.where(
    (data["study_condition"] == "control") & (data["BMI"] >= 18.5) & (data["BMI"] < 25), 1, 0
)

# Filter bacteria columns
bacteria_columns = [col for col in data.columns if '|' in col or 'healthy' in col]
data = data[bacteria_columns]

# Modify bacteria columns
modified_bacteria_columns = [col if col == 'healthy' else 's__' + col.split('s__', 1)[-1] for col in data.columns]
data.columns = modified_bacteria_columns

# Load fifty species
with open("50_species.txt") as f:
    fifty_species = [line.strip() for line in f]

# Filter data to include matching columns
matching_columns = [col for col in data.columns if col in fifty_species or col == 'healthy']
data = data[matching_columns]

# Split data into features and target
X = data.drop(columns='healthy')
y = data['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

# Apply CLR transformation
X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert the CLR transformed data back to DataFrame
X_train_clr = pd.DataFrame(X_train_clr, index=X_train.index, columns=X_train.columns)
X_val_clr = pd.DataFrame(X_val_clr, index=X_val.index, columns=X_val.columns)
X_test_clr = pd.DataFrame(X_test_clr, index=X_test.index, columns=X_test.columns)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_clr, y_train)

# Save the data
datasets = {
    'Training set': (X_train_smote, y_train_smote),
    'Validation set': (X_val_clr, y_val),
    'Test set': (X_test_clr, y_test)
}

output_dir = root / 'data' / '50SPC_CLR_smote'
output_dir.mkdir(parents=True, exist_ok=True)

# Write dataset distribution
with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([pd.DataFrame(X).reset_index(drop=True), pd.DataFrame(y).reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)

# %% [markdown]
# # CLR PCA LowAbFilt

# %%
# Load data
root = Path.cwd().parents[1]
full = root / "data" / "raw" / "full_df_with_meta.csv"
full_df_with_meta = pd.read_csv(full)

full_df_with_meta = full_df_with_meta.dropna(subset=['study_condition'])

# remove carcinoma_surgery_history
full_df_with_meta = full_df_with_meta[full_df_with_meta['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
full_df_with_meta['healthy'] = np.where(
    (full_df_with_meta["study_condition"] == "control") & (full_df_with_meta["BMI"] >= 18.5) & (full_df_with_meta["BMI"] < 25), 1, 0
)



# %%
# Get shape of full_df_with_meta
print(f"The number of sample before removing subjects with carcinoma surgery history: {full_df_with_meta.shape}")  # (8261,1585)
# remove the samples with study_condition == carcinoma_surgery_history
full_df_with_meta = full_df_with_meta[full_df_with_meta['study_condition'] != "carcinoma_surgery_history"]
print(f"The number of sample after removing subjects with carcinoma surgery history: {full_df_with_meta.shape}")  # (8221,1585)
print(full_df_with_meta['study_condition'].value_counts())  # 5620 healthy controls, 2641 patients

# get number of unique subject_id
print(f"The number of unique subjects in the data are {len(full_df_with_meta.subject_id.unique())}")  # 8221

# save a .txt file with all the unique values in the study_condition column
with open('study_condition_values.txt', 'w') as f:
    for item in full_df_with_meta['study_condition'].unique():
        f.write("%s\n" % item)

# Make a new column called 'healthy' that is 1 if study_condition == control, 0 if study_condition is not control
full_df_with_meta['healthy'] = np.where(full_df_with_meta["study_condition"] == "control", 1, 0)

disease_counts = full_df_with_meta['study_condition'].value_counts().reset_index()
disease_counts.columns = ['Disease', 'Count']

# Create the structure for the CSV
disease_info = pd.DataFrame({
    'Name': disease_counts['Disease'],
    'Abbreviation': '',
    'Count': disease_counts['Count'],
    'Description': '',
    'Diagnosis': ''
})

# Save the DataFrame to a CSV file
disease_info.to_csv('disease_info.csv', index=False)

# %%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
root = Path.cwd().parents[1]
full = root / "data" / "raw" / "full_df_with_meta.csv"
full_df_with_meta = pd.read_csv(full)

full_df_with_meta = full_df_with_meta.dropna(subset=['study_condition'])

# Remove carcinoma_surgery_history
full_df_with_meta = full_df_with_meta[full_df_with_meta['study_condition'] != "carcinoma_surgery_history"]

# Create 'healthy' column if person is flagged as control AND has a BMI within the healthy range
full_df_with_meta['healthy'] = np.where(
    (full_df_with_meta["study_condition"] == "control") & (full_df_with_meta["BMI"] >= 18.5) & (full_df_with_meta["BMI"] < 25), 1, 0
)

# Get shape of full_df_with_meta
print(f"The number of sample before removing subjects with carcinoma surgery history: {full_df_with_meta.shape}")  # (8261,1585)
# Remove the samples with study_condition == carcinoma_surgery_history
full_df_with_meta = full_df_with_meta[full_df_with_meta['study_condition'] != "carcinoma_surgery_history"]
print(f"The number of sample after removing subjects with carcinoma surgery history: {full_df_with_meta.shape}")  # (8221,1585)
print(full_df_with_meta['study_condition'].value_counts())  # 5620 healthy controls, 2641 patients

# Get number of unique subject_id
print(f"The number of unique subjects in the data are {len(full_df_with_meta.subject_id.unique())}")  # 8221

# Save a .txt file with all the unique values in the study_condition column
with open('study_condition_values.txt', 'w') as f:
    for item in full_df_with_meta['study_condition'].unique():
        f.write("%s\n" % item)

disease_counts = full_df_with_meta['study_condition'].value_counts().reset_index()
disease_counts.columns = ['Disease', 'Count']

# Create the structure for the CSV
disease_info = pd.DataFrame({
    'Name': disease_counts['Disease'],
    'Abbreviation': '',
    'Count': disease_counts['Count'],
    'Description': '',
    'Diagnosis': ''
})

# Save the DataFrame to a CSV file
disease_info.to_csv('disease_info.csv', index=False)

# Define all bacteria columns as containing the string "|"
bacteria_columns = [col for col in full_df_with_meta.columns if '|' in col]

# Split data into features (X) and target (y)
X = full_df_with_meta[bacteria_columns]
y = full_df_with_meta['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.15, 
                                                    random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=X_test.shape[0] / X_train.shape[0],  
                                                  random_state=42,stratify=y_train)

# Calculate the sum for each column before CLR
species_columns = [col for col in X_train.columns if '|' in col]
columns_sums = X_train[species_columns].sum()
total_sum = columns_sums.sum()
columns_percentages = (columns_sums / total_sum) * 100

# Find columns where percentage is 0.1% or more
filtered_columns = columns_percentages[columns_percentages >= 0.1].index

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

# Apply CLR transformation
X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert back to pandas DataFrame
X_train_clr_df = pd.DataFrame(X_train_clr, columns=X_train.columns)
X_val_clr_df = pd.DataFrame(X_val_clr, columns=X_val.columns)
X_test_clr_df = pd.DataFrame(X_test_clr, columns=X_test.columns)

# Filter X_train_clr_df, X_val_clr_df, and X_test_clr_df to include only the relevant columns
X_train_filtered = X_train_clr_df[filtered_columns]
X_val_filtered = X_val_clr_df[filtered_columns]
X_test_filtered = X_test_clr_df[filtered_columns]

# Perform PCA
pca_model = PCA(n_components=0.95)
X_train_pca = pca_model.fit_transform(X_train_filtered)
X_val_pca = pca_model.transform(X_val_filtered)
X_test_pca = pca_model.transform(X_test_filtered)

# Plot cumulative explained variance
cumulative_explained_variance = np.cumsum(pca_model.explained_variance_ratio_)
print(f"Cumulative explained variance is: {cumulative_explained_variance}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('PCA: Number of Principal Components')
plt.ylabel('PCA: Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Scree plot of explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), pca_model.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('PCA: Principal Component')
plt.ylabel('PCA: Explained Variance Ratio')
plt.title('PCA: Scree Plot')
plt.grid(True)
plt.show()

# Save the PCA-transformed data
datasets = {
    'Training set': (X_train_pca, y_train), 
    'Validation set': (X_val_pca, y_val), 
    'Test set': (X_test_pca, y_test)
}
output_dir = root / 'data' / 'CLR_PCA_LowAbFilt'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([pd.DataFrame(X), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)


# %% [markdown]
# # CLR PCA LowAbFilt undersampled

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
root = Path.cwd().parents[1]
full = root / "data" / "raw" / "full_df_with_meta.csv"
full_df_with_meta = pd.read_csv(full)

full_df_with_meta = full_df_with_meta.dropna(subset=['study_condition'])

full_df_with_meta['healthy'] = np.where(
    (full_df_with_meta["study_condition"] == "control") & (full_df_with_meta["BMI"] >= 18.5) & (full_df_with_meta["BMI"] < 25), 1, 0)


# Define all bacteria columns as containing the string "|"
bacteria_columns = [col for col in full_df_with_meta.columns if '|' in col]

# Split data into features (X) and target (y)
X = full_df_with_meta[bacteria_columns]
y = full_df_with_meta['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Identify the minority class in the training set
minority_class = y_train.value_counts().idxmin()
minority_count = y_train.value_counts().min()

# Separate the majority and minority classes
X_train_minority = X_train[y_train == minority_class]
y_train_minority = y_train[y_train == minority_class]

X_train_majority = X_train[y_train != minority_class]
y_train_majority = y_train[y_train != minority_class]

# Undersample the majority class
X_train_majority_downsampled = X_train_majority.sample(n=minority_count, random_state=42)
y_train_majority_downsampled = y_train_majority.sample(n=minority_count, random_state=42)

# Combine the downsampled majority class with the minority class
X_train_balanced = pd.concat([X_train_minority, X_train_majority_downsampled])
y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])

# Calculate the sum for each column before CLR
species_columns = [col for col in X_train_balanced.columns if '|' in col]
columns_sums = X_train_balanced[species_columns].sum()
total_sum = columns_sums.sum()
columns_percentages = (columns_sums / total_sum) * 100

# Find columns where percentage is 0.1% or more
filtered_columns = columns_percentages[columns_percentages >= 0.1].index

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train_balanced + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert back to pandas DataFrame
X_train_clr_df = pd.DataFrame(X_train_clr, columns=X_train.columns)
X_val_clr_df = pd.DataFrame(X_val_clr, columns=X_val.columns)
X_test_clr_df = pd.DataFrame(X_test_clr, columns=X_test.columns)

# Filter X_train_clr_df, X_val_clr_df, and X_test_clr_df to include only the relevant columns
X_train_filtered = X_train_clr_df[filtered_columns]
X_val_filtered = X_val_clr_df[filtered_columns]
X_test_filtered = X_test_clr_df[filtered_columns]

# Perform PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_filtered)
X_val_pca = pca.transform(X_val_filtered)
X_test_pca = pca.transform(X_test_filtered)

# Plot cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative explained variance is: {cumulative_explained_variance}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('PCA: Number of Principal Components')
plt.ylabel('PCA: Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Scree plot of explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('PCA: Principal Component')
plt.ylabel('PCA: Explained Variance Ratio')
plt.title('PCA: Scree Plot')
plt.grid(True)
plt.show()

# Save the PCA-transformed data
datasets = {
    'Training set': (X_train_pca, y_train_balanced), 
    'Validation set': (X_val_pca, y_val), 
    'Test set': (X_test_pca, y_test)
}
output_dir = root / 'data' / 'CLR_PCA_LowAbFilt_undersampled'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([pd.DataFrame(X), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
root = Path.cwd().parents[1]
full = root / "data" / "raw" / "full_df_with_meta.csv"
full_df_with_meta = pd.read_csv(full)

full_df_with_meta = full_df_with_meta.dropna(subset=['study_condition'])

# Make a new column called 'healthy' that is 1 if study_condition == control, 0 if study_condition is not control
full_df_with_meta['healthy'] = np.where(
    (full_df_with_meta["study_condition"] == "control") & (full_df_with_meta["BMI"] >= 18.5) & (full_df_with_meta["BMI"] < 25), 1, 0)


# Define all bacteria columns as containing the string "|"
bacteria_columns = [col for col in full_df_with_meta.columns if '|' in col]

# Split data into features (X) and target (y)
X = full_df_with_meta[bacteria_columns]
y = full_df_with_meta['healthy']

# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0] / X_train.shape[0], random_state=42, stratify=y_train)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Calculate the sum for each column before CLR
species_columns = [col for col in X_train_smote.columns if '|' in col]
columns_sums = X_train_smote[species_columns].sum()
total_sum = columns_sums.sum()
columns_percentages = (columns_sums / total_sum) * 100

# Find columns where percentage is 0.1% or more
filtered_columns = columns_percentages[columns_percentages >= 0.1].index

# Apply pseudocount to avoid zeros
pseudocount = 1e-6
X_train_pseudocount = X_train_smote + pseudocount
X_val_pseudocount = X_val + pseudocount
X_test_pseudocount = X_test + pseudocount

X_train_clr = clr(X_train_pseudocount)
X_val_clr = clr(X_val_pseudocount)
X_test_clr = clr(X_test_pseudocount)

# Convert back to pandas DataFrame
X_train_clr_df = pd.DataFrame(X_train_clr, columns=X_train.columns)
X_val_clr_df = pd.DataFrame(X_val_clr, columns=X_val.columns)
X_test_clr_df = pd.DataFrame(X_test_clr, columns=X_test.columns)

# Filter X_train_clr_df, X_val_clr_df, and X_test_clr_df to include only the relevant columns
X_train_filtered = X_train_clr_df[filtered_columns]
X_val_filtered = X_val_clr_df[filtered_columns]
X_test_filtered = X_test_clr_df[filtered_columns]

# Perform PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_filtered)
X_val_pca = pca.transform(X_val_filtered)
X_test_pca = pca.transform(X_test_filtered)

# Plot cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative explained variance is: {cumulative_explained_variance}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('PCA: Number of Principal Components')
plt.ylabel('PCA: Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Scree plot of explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('PCA: Principal Component')
plt.ylabel('PCA: Explained Variance Ratio')
plt.title('PCA: Scree Plot')
plt.grid(True)
plt.show()

# Save the PCA-transformed data
datasets = {
    'Training set': (X_train_pca, y_train_smote), 
    'Validation set': (X_val_pca, y_val), 
    'Test set': (X_test_pca, y_test)
}
output_dir = root / 'data' / 'CLR_PCA_LowAbFilt_smote'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dataset_distribution.txt', 'w') as file:
    header = "{:<15} | {:>4} | {:>4} | {:>7}".format("Dataset", "1", "0", "Rows")
    file.write(header + '\n')
    
    for name, (X, y) in datasets.items():
        proportion = y.value_counts(normalize=True)
        total = len(y)
        proportion_1 = proportion.get(1, 0)
        proportion_0 = proportion.get(0, 0)
        line = "{:<15} | {:.2f} | {:.2f} | {:>7}".format(name, proportion_1, proportion_0, total)
        file.write(line + '\n')

# Save CSV files with the specified names
csv_names = ['train.csv', 'val.csv', 'test.csv']
for (X, y), csv_name in zip(datasets.values(), csv_names):
    combined_data = pd.concat([pd.DataFrame(X), y.reset_index(drop=True)], axis=1)
    combined_data.to_csv(output_dir / csv_name, index=False)




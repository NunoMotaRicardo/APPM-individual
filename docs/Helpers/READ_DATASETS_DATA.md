# Read Datasets Data Function

## Overview

The ead_datasets_data function reads previously saved datasets from a test folder and returns them in the same format as the creation functions (create_resampled_datasets and create_stratified_datasets).

## Function Signature

```python
def read_datasets_data(test_folder_path):
    """
    Read datasets from a test folder
    
    Parameters:
    - test_folder_path: path to the test folder containing datasets
    
    Returns:
    - datasets_info: dict with keys 'dataset_1', 'dataset_2', etc., containing metadata
    - datasets_data: list of DataFrames
    """
```

## Parameters

### test_folder_path (str)
- **Required**: Yes
- **Description**: Absolute or relative path to the test folder containing the datasets
- **Example**: 'data/selection3/test-1' or 'C:/path/to/data/selection3/test-1'

## Returns

The function returns a tuple of two elements:

### 1. datasets_info (dict)
A dictionary with dataset metadata:
- **Keys**: 'dataset_1', 'dataset_2', ..., 'dataset_N'
- **Values**: Dictionary containing:
  - 'start_date': Start date of the dataset (str, format 'YYYY-MM-DD')
  - 'end_date': End date of the dataset (str, format 'YYYY-MM-DD')
  - 'selected_symbols': List of ticker symbols in the dataset (list of str)
  - 'assets': DataFrame with asset information (only for stratified datasets, optional)

### 2. datasets_data (list)
A list of pandas DataFrames, one for each dataset:
- **Index**: DateTime index with dates
- **Columns**: Multi-level columns with:
  - Level 0: Ticker symbols
  - Level 1: Price types ('open', 'high', 'low', 'close', 'adjusted', 'volume')

## Usage Examples

### Basic Usage

```python
from Backtester.helpers import read_datasets_data
import os

# Define test folder path
test_path = os.path.join(os.getcwd(), 'data', 'selection3', 'test-1')

# Read datasets
datasets_info, datasets_data = read_datasets_data(test_path)

print(f"Loaded {len(datasets_data)} datasets")
print(f"First dataset shape: {datasets_data[0].shape}")
print(f"First dataset date range: {datasets_info['dataset_1']['start_date']} to {datasets_info['dataset_1']['end_date']}")
```

### Accessing Specific Dataset

```python
# Access first dataset
dataset_1 = datasets_data[0]
dataset_1_info = datasets_info['dataset_1']

print(f"Dataset 1 has {len(dataset_1_info['selected_symbols'])} symbols")
print(f"Symbols: {dataset_1_info['selected_symbols']}")

# Access specific symbol data
symbol = 'SPY'
spy_data = dataset_1[symbol]
print(spy_data.head())
```

### Iterating Through All Datasets

```python
for i, (dataset_df, dataset_key) in enumerate(zip(datasets_data, datasets_info.keys()), 1):
    info = datasets_info[dataset_key]
    print(f"Dataset {i}:")
    print(f"  Date range: {info['start_date']} to {info['end_date']}")
    print(f"  Shape: {dataset_df.shape}")
    print(f"  Symbols: {len(info['selected_symbols'])}")
```

### Working with Stratified Datasets

If the datasets were created using create_stratified_datasets, they will have asset class information:

```python
datasets_info, datasets_data = read_datasets_data(test_path)

# Check if first dataset has asset information
if 'assets' in datasets_info['dataset_1']:
    assets_df = datasets_info['dataset_1']['assets']
    print(assets_df.head())
    
    # Group by asset class
    if 'asset_class' in assets_df.columns:
        print(assets_df.groupby('asset_class').size())
```

## File Structure Expected

The function expects the following file structure in the test folder:

```
test_folder_path/
├── datasets_info.json          # Metadata for all datasets
├── datasets_settings.json       # Settings used to create datasets (not read by this function)
├── test_settings.json          # Test settings (not read by this function)
└── datasets/
    ├── dataset_1.csv
    ├── dataset_2.csv
    ├── dataset_3.csv
    └── ...
```

## Error Handling

The function will raise exceptions in the following cases:

1. **FileNotFoundError**: If datasets_info.json is not found:
   ```python
   FileNotFoundError: datasets_info.json not found in {test_folder_path}
   ```

2. **FileNotFoundError**: If datasets/ folder is not found:
   ```python
   FileNotFoundError: datasets folder not found in {test_folder_path}
   ```

3. **Warning**: If a specific dataset CSV file is missing (continues with remaining datasets):
   ```
   Warning: {dataset_file} not found, skipping...
   ```

## Notes

1. **Dataset Order**: Datasets are loaded in numerical order (dataset_1, dataset_2, ..., dataset_10, dataset_11, ...) regardless of how they appear in the file system.

2. **Multi-level Columns**: The CSV files are expected to have multi-level column headers (Ticker and Price type).

3. **Date Index**: The index is automatically parsed as dates.

4. **Asset Information**: If datasets were created with create_stratified_datasets and contain asset information as dictionaries in the JSON, they will be converted back to pandas DataFrames.

5. **Compatibility**: The function is designed to work with datasets created by both create_resampled_datasets and create_stratified_datasets.

## Related Functions

- create_resampled_datasets: Creates random resampled datasets
- create_stratified_datasets: Creates stratified datasets based on asset classes
- prepare_folders: Prepares folder structure for storing datasets

## See Also

- [BacktestResults User Guide](../Results/BacktestResults_UserGuide.md)
- [Custom Parameters Implementation](../CustomParams/CUSTOM_PARAMS_IMPLEMENTATION.md)

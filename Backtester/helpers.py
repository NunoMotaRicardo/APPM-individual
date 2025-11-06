import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime, timedelta


def create_resampled_datasets(master_dataset, num_assets=20, backtest_duration=504, num_datasets=20, random_seed=None):
    """
    Create resampled datasets from financial data
    
    Parameters:
    - data_dict: dictionary with 'adjusted' key containing price data
    - num_assets: number of assets per dataset
    - backtest_duration: number of time periods per dataset
    - num_datasets: number of datasets to create
    - random_seed: random seed for reproducibility
    
    Returns:
    - datasets_info: dict with keys 'dataset_1', 'dataset_2', etc., containing metadata
    - datasets_data: list of DataFrames
    """
    #np.random.seed(random_seed)
    if random_seed is not None:
        random.seed(random_seed)
    
    #master_dataset = data_dict['Adj Close'].dropna(how='all')
    #all_symbols = list(master_dataset.columns)
    all_symbols = list(master_dataset.columns.get_level_values(0).unique())
    datasets_data = []
    datasets_info = {}
    
    for i in range(num_datasets):
        # Randomly select num_assets stocks
        selected_symbols = random.sample(all_symbols, min(num_assets, len(all_symbols)))

        max_start_idx = len(master_dataset) - backtest_duration
        if max_start_idx > 0:
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + backtest_duration
            
            # Extract the subset
            subset_data = master_dataset[selected_symbols].iloc[start_idx:end_idx]
            
            # Store in dataset
            datasets_data.append(subset_data)
            datasets_info[f'dataset_{i+1}'] = {
                'start_date': subset_data.index.min().strftime('%Y-%m-%d') if not subset_data.empty else None,
                'end_date': subset_data.index.max().strftime('%Y-%m-%d') if not subset_data.empty else None,
                'selected_symbols': selected_symbols,
            }    
    return datasets_info, datasets_data


def prepare_folders(universe_name, backtest_duration:int, lookback_periods:int=0, num_datasets:int=10, random_seed:int=None,num_assets:int=None):
    test_settings = {
        "universe_name": universe_name,
        "backtest_duration": backtest_duration,
        "lookback_periods": lookback_periods,
        "num_datasets": num_datasets,
        "random_seed": random_seed,
        "num_assets": num_assets
    }
    universe_folder_path = os.path.join(os.getcwd(),'data', universe_name)

    universe_data = pd.read_csv(os.path.join(universe_folder_path, f"universe_data.csv"), header=[0,1], index_col=0, parse_dates=True)
    if isinstance(universe_data.columns, pd.MultiIndex):
        asset_names = universe_data.columns.get_level_values(0).unique()
        print(f"Downloaded data for {len(asset_names)} assets")
    else:
        print(f"Downloaded data for {len(universe_data.columns)} assets")
    print(f"Universe data date range: {universe_data.index.min().strftime('%Y-%m-%d')} to {universe_data.index.max().strftime('%Y-%m-%d')}")

    universe_info_dict = json.load(open(os.path.join(universe_folder_path, f"universe_settings.json"), 'r'))
    universe_symbols = universe_info_dict['universe_symbols']
    if ('universe_asset_class' in universe_info_dict) and (len(universe_info_dict['universe_asset_class']) == len(universe_symbols)):
        universe_asset_classes = universe_info_dict['universe_asset_class']
        universe_info_df = pd.DataFrame({'symbol': universe_symbols, 'asset_class': universe_asset_classes})
    else:
        universe_info_df = pd.DataFrame({'symbol': universe_symbols})
    universe_info_df = universe_info_df.set_index('symbol')

    # Check if any folder starting with 'test' exists in the universe_name directory
    # Give a name to the test based on the existing folders
    # Defaults to test-1
    test_folders = [f for f in os.listdir(universe_folder_path) if os.path.isdir(os.path.join(universe_folder_path, f)) and f.startswith('test')]
    test_name = 'test-1'
    if test_folders:
        # Extract ordinals and find the greatest one
        ordinals = []
        for folder in test_folders:
            parts = folder.split('-')
            if len(parts) == 2 and parts[0] == 'test':
                ordinals.append(int(parts[1]))
            max_ordinal = max(ordinals)
            test_name = f'test-{max_ordinal + 1}'
            test_folder_path = os.path.join(universe_folder_path, test_name)
            os.makedirs(test_folder_path, exist_ok=True)
            print(f"Created folder: {test_folder_path}")
    else:
        test_folder_path = os.path.join(universe_folder_path, test_name)
        os.makedirs(test_folder_path, exist_ok=True)
        print(f"Created folder: {test_folder_path}")

    test_settings['test_name'] = test_name
    test_settings['test_folder_path'] = test_folder_path

    ### Save settings to JSON file

    with open(os.path.join(test_folder_path, 'test_settings.json'), 'w') as f:
        json.dump(test_settings, f, indent=4)

    return universe_info_df,universe_data, test_settings


def create_stratified_datasets(master_dataset, universe_info_df:pd.DataFrame, group_by='asset_class', assets_per_group:int=1, backtest_duration=504, num_datasets=2, random_seed=None):
    #np.random.seed(random_seed)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    #master_symbols = list(master_dataset.columns.get_level_values(0).unique())
    ## Remove all symbols from universe_info_df that are not in master_symbols
    #filtered_symbols = [s for s in universe_info_df.index if s in master_symbols]
    #universe_info_df = universe_info_df.loc[filtered_symbols]

    datasets_data = []
    datasets_info = {}
    for i in range(num_datasets):
        # Stratified sampling: randomly select one symbol per asset class
        sampled_assets = []
        for asset_class, group in universe_info_df.groupby(group_by):
            symbols_in_class = group.index.tolist()
            if len(symbols_in_class) >= assets_per_group:
                sampled = random.sample(symbols_in_class, assets_per_group)
            else:
                # If not enough assets, sample with replacement
                print(f"Warning: Not enough assets in class {asset_class} to sample {assets_per_group}. Sampling all available assets.")
                sampled = symbols_in_class
            sampled_assets.extend(sampled)
        # Ensure sampled_assets are in master_symbols and unique
        #sampled_assets = [s for s in sampled_assets if s in master_symbols]
        sampled_assets = list(dict.fromkeys(sampled_assets))  # preserve order, remove duplicates
        dataset_info_df = universe_info_df.loc[sampled_assets].copy()

        # Extract columns for sampled assets (all price types)
        if isinstance(master_dataset.columns, pd.MultiIndex):
            cols = [col for col in master_dataset.columns if col[0] in sampled_assets]
        else:
            cols = [col for col in master_dataset.columns if col in sampled_assets]
        sampled_data = master_dataset[cols].copy()

        # Optionally sample time window
        if sampled_data.shape[0] > backtest_duration:
            start_idx = random.randint(0, sampled_data.shape[0] - backtest_duration)
            sampled_data = sampled_data.iloc[start_idx:start_idx + backtest_duration]
            sampled_data.fillna(0, inplace=True)

        datasets_data.append(sampled_data)
        datasets_info[f'dataset_{i+1}'] = {
            'start_date': str(sampled_data.index.min().date()),
            'end_date': str(sampled_data.index.max().date()),
            'assets': dataset_info_df,
        }
        print(f"\n{datasets_info}")

    return datasets_info, datasets_data


def read_datasets_data(test_folder_path):
    """
    Read datasets from a test folder
    
    Parameters:
    - test_folder_path: path to the test folder containing datasets
    
    Returns:
    - datasets_info: dict with keys 'dataset_1', 'dataset_2', etc., containing metadata
    - datasets_data: list of DataFrames
    """
    
    # Read datasets_info.json
    datasets_info_path = os.path.join(test_folder_path, 'datasets_info.json')
    if not os.path.exists(datasets_info_path):
        raise FileNotFoundError(f"datasets_info.json not found in {test_folder_path}")
    
    with open(datasets_info_path, 'r') as f:
        datasets_info = json.load(f)
    
    # Read each dataset CSV file
    datasets_data = []
    datasets_folder = os.path.join(test_folder_path, 'datasets')
    
    if not os.path.exists(datasets_folder):
        raise FileNotFoundError(f"datasets folder not found in {test_folder_path}")
    
    # Sort dataset keys to maintain order (dataset_1, dataset_2, ..., dataset_10, ...)
    dataset_keys = sorted(datasets_info.keys(), key=lambda x: int(x.split('_')[1]))
    
    for dataset_key in dataset_keys:
        dataset_file = os.path.join(datasets_folder, f"{dataset_key}.csv")
        
        if not os.path.exists(dataset_file):
            print(f"Warning: {dataset_file} not found, skipping...")
            continue
        
        # Read the CSV with multi-level columns (Ticker and Price)
        df = pd.read_csv(dataset_file, header=[0, 1], index_col=0, parse_dates=True)
        datasets_data.append(df)
        
        # Convert assets info back to DataFrame if it was saved as dict/list
        if 'assets' in datasets_info[dataset_key]:
            assets_data = datasets_info[dataset_key]['assets']
            if isinstance(assets_data, dict):
                # Convert dict to DataFrame
                datasets_info[dataset_key]['assets'] = pd.DataFrame(assets_data)
    
    print(f"Loaded {len(datasets_data)} datasets from {test_folder_path}")
    
    return datasets_info, datasets_data

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        return obj.item()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, 'strftime'):  # This catches datetime.date and other date-like objects
        return obj.strftime('%Y-%m-%d')   
    elif isinstance(obj, (list, dict, str, int, float, bool)):
        return obj
    else:
        return type(obj).__name__ 

def recursive_serialize(data):
    if isinstance(data, dict):
        return {k: recursive_serialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_serialize(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_serialize(v) for v in data)
    else:
        return make_serializable(data)
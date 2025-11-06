import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import os

def get_yahoo_data(tickers, start_date, end_date, interval='1d'):
    # Download data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, group_by='ticker', auto_adjust=False, threads=True)
    
    # If only one ticker, the columns won't be multi-indexed, so we handle that case
    #if len(tickers) == 1:
    #    ticker = tickers[0]
    #    data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
    
    return data

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
    - dict of datasets
    """
    #np.random.seed(random_seed)
    if random_seed is not None:
        random.seed(random_seed)
    
    #master_dataset = data_dict['Adj Close'].dropna(how='all')
    #all_symbols = list(master_dataset.columns)
    all_symbols = list(master_dataset.columns.get_level_values(0).unique())
    datasets_data = []
    datasets_info = []
    
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
            datasets_info.append({
                'start_date': subset_data.index.min().strftime('%Y-%m-%d') if not subset_data.empty else None,
                'end_date': subset_data.index.max().strftime('%Y-%m-%d') if not subset_data.empty else None,
                'selected_symbols': selected_symbols,
            })    
    return datasets_info, datasets_data


def extract_returns_by_strategy(bt_backtest):
    results_out = {}
    
    for strat_name,datasets in bt_backtest.items():
        results_out[strat_name] = {}
        for ds_name, res in datasets.items():
            rets = res.get('returns', None)
            dates = res.get('return_dates', None)
            
            # Normalize returns to list
            if rets is None:
                rets_list = []
            else:
                rets_list = np.asarray(rets).tolist()
            
            # If dates missing or length mismatch, try to recover from stored dataset index
            if dates is None or len(dates) != len(rets_list):
                ds_obj = None
                if hasattr(bt_backtest, 'datasets'):
                    ds_obj = bt_backtest.datasets.get(ds_name, None)
                if ds_obj is None:
                    # fallback to empty or synthetic daily dates ending at end_date
                    if len(rets_list) > 0:
                        dates_list = pd.date_range(end=pd.to_datetime(end_date), periods=len(rets_list), freq='D').strftime('%Y-%m-%d').tolist()
                    else:
                        dates_list = []
                else:
                    idx = pd.to_datetime(ds_obj.index)
                    # If index length matches returns use it, otherwise try to align end of index to returns length
                    if len(idx) == len(rets_list):
                        dates_list = idx.strftime('%Y-%m-%d').tolist()
                    elif len(idx) >= len(rets_list) and len(rets_list) > 0:
                        # align latest returns to the tail of the dataset index
                        dates_list = idx[-len(rets_list):].strftime('%Y-%m-%d').tolist()
                    elif len(rets_list) > 0:
                        dates_list = pd.date_range(start=idx.min(), periods=len(rets_list), freq='D').strftime('%Y-%m-%d').tolist()
                    else:
                        dates_list = []
            else:
                dates_list = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
            
            results_out[strat_name][ds_name] = {
                'dates': dates_list,
                'returns': rets_list
            }
    return results_out

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
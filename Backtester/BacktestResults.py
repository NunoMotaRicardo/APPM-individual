import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

class TestResults:
    def __init__(self, test_path: str | Path):
        self.test_path = Path(test_path).resolve()
        if not self.test_path.exists():
            raise FileNotFoundError(f'Test path not found: {self.test_path}')
        self.results_path = self.test_path / 'results'

        self.test_settings = self._load_json(self.test_path / 'test_settings.json', label='test settings')
        self.universe_name = self.test_settings.get('universe_name', '')
        self.name= self.test_settings.get('test_name', self.test_path.stem)
        self.num_assets = self.test_settings.get('num_assets', None)
        self.lookback_periods = int(self.test_settings.get('lookback_periods', None))
        self.duration = int(self.test_settings.get('backtest_duration', None))
        self.random_seed = int(self.test_settings.get('random_seed', None))
        self.num_datasets = int(self.test_settings.get('num_datasets', None))
        self.universe_settings = self._load_universe_settings()
        self.universe_data_path = self.test_path.parent / 'universe_data.csv'
        self._load_datasets()
        self._load_strategies()


    def _load_universe_settings(self) -> Dict[str, Any]:
        universe_settings = self._load_json((self.test_path.parent / 'universe_settings.json').resolve(), label='universe settings', missing_ok=True)
        # Create a DataFrame with index as universe_symbols and column as universe_asset_class
        symbols = universe_settings.get('universe_symbols', [])
        asset_class = universe_settings.get('universe_asset_class', None)
        universe_settings['universe_symbols'] = pd.DataFrame(
            index=pd.Index(symbols, name='symbol'),
            data={
                'universe_asset_class': asset_class,
            }
        )
        universe_settings.pop('universe_asset_class', None)
        universe_info = pd.read_csv((self.test_path.parent / 'universe_info.csv').resolve(), index_col=0)

        merged = universe_settings['universe_symbols'].join(
            universe_info[['shortName', 'quoteType', 'symbol', 'language', 'region','exchange', 'currency']],
            how='left'
        )
        universe_settings['universe_symbols'] = merged

        return universe_settings

    def _load_datasets(self) -> None:
        datasets_info = self._load_json(
            self.test_path / 'datasets_info.json',
            label='datasets info'
        )
        datasets_data_files = (self.test_path / 'datasets/').resolve()
        dataset_files = sorted([
            f for f in datasets_data_files.glob('*')
            if f.is_file()
        ])
        self.datasets = {}
        for i, dataset_file in enumerate(dataset_files):

            dataset_file_name=dataset_file.stem

            self.datasets[dataset_file_name] = {}
            self.datasets[dataset_file_name]['name'] = dataset_file.stem
            self.datasets[dataset_file_name]['path'] = dataset_file
            self.datasets[dataset_file_name]['start_date'] = datasets_info[dataset_file_name].get('start_date')
            self.datasets[dataset_file_name]['end_date'] = datasets_info[dataset_file_name].get('end_date')
            assets = datasets_info[dataset_file_name].get('assets', [])
            self.datasets[dataset_file_name]['symbols'] = pd.DataFrame(assets)


    def _load_strategies(self) -> None:
        strategies_path = self.results_path
        if not strategies_path.exists() or not strategies_path.is_dir():
            self.strategies = {}
            return
        strategy_files = sorted([
            f for f in strategies_path.glob('*.json')
            if f.is_file()
        ])
        self.strategies = {}
        for strategy_file in strategy_files:
            payload = self._load_json(strategy_file, label=f'strategy file {strategy_file.name}', missing_ok=True)
            strategy_data = StrategyResults(payload,self.lookback_periods)
            self.strategies[strategy_data.name] = strategy_data

    def _load_json(self, path: Path, label: str = 'file', missing_ok: bool = False) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        except FileNotFoundError:
            if missing_ok:
                return {}
            raise FileNotFoundError(f'Cannot locate {label} at {path}') from None
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON in {label}: {path}') from exc

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())
    
    def get_datasets_info(self,dataset_names:list=[]):
        if not dataset_names:
            return self.datasets.copy()
        return {name: self.datasets.get(name, {}) for name in dataset_names}

    def get_datasets_data(self,datasets_filter_list:list=[],column:str=""):
        datasets_list = self.list_datasets()
        if not datasets_filter_list:
            datasets_filter_list = datasets_list
        data = {}
        for dataset_name in datasets_filter_list:
            dataset_data = pd.read_csv(self.datasets[dataset_name]['path'], index_col=0, header=[0, 1])
            if column!="":
                dataset_data = dataset_data.xs(column, level=1, axis=1)
            data[dataset_name] = dataset_data
            data[dataset_name] = data[dataset_name].iloc[self.lookback_periods:]
        return data
    
    def list_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def get_strategies_stats(self,strategy_names:list=[]):
        if not strategy_names:
            strategy_names = self.list_strategies()
        performance = pd.DataFrame()
        for strategy_name in strategy_names:
            strategy_performance = self.strategies.get(strategy_name).get_performance()
            if strategy_performance is not None and not strategy_performance.empty:
                performance = pd.concat([performance, strategy_performance], axis=0)
        return performance

    def get_strategies_bt_performance(self,strategy_names:list=[],aggregator:str="mean"):
        if not strategy_names:
            strategy_names = self.list_strategies()
        performance = pd.DataFrame()
        for strategy_name in strategy_names:
            strategy_performance = self.strategies.get(strategy_name).get_datasets_bt_performance().copy()
            strategy_performance = strategy_performance.fillna(0)  # Use non-inplace version to avoid issues
            #print (strategy_performance)
            if aggregator=="mean":
                strategy_performance = strategy_performance.mean(axis=0).to_frame().T
            elif aggregator=="median":
                strategy_performance = strategy_performance.median(axis=0).to_frame().T
            elif aggregator=="max":
                strategy_performance = strategy_performance.max(axis=0).to_frame().T
            elif aggregator=="min":
                strategy_performance = strategy_performance.min(axis=0).to_frame().T
            strategy_performance = strategy_performance.set_index(pd.Index([strategy_name]))
            if strategy_performance is not None and not strategy_performance.empty:
                performance = pd.concat([performance, strategy_performance], axis=0)
        return performance

class StrategyResults:
    def __init__(self,payload: Dict,lookback_periods: int):

        self.name: str = payload.get('Strategy', '')

        datasets_payload = payload.get('Datasets', {})
        self.datasets = {}
        for dataset_name, dataset_payload in datasets_payload.items():
            dataset_result = DatasetResults(
                name=dataset_name,
                payload=dataset_payload,
                lookback_periods=lookback_periods,
                strategy_name=self.name
            )
            self.datasets[dataset_name] = dataset_result

    def _compute_performance(self) -> pd.Series:
        returns = self.get_returns()
        if returns.empty:
            return pd.Series(name=self.name, dtype=float)
        return_series = returns['return']
        cumulative_log_returns = np.log1p(return_series).cumsum()
        stats = pd.Series({
            'annual_return': np.expm1(cumulative_log_returns.iloc[-1]) * (252 / return_series.count()) if return_series.count() > 0 else np.nan,
            'annual_volatility': return_series.std(ddof=1) * np.sqrt(252) if return_series.count() > 1 else np.nan,
            'max_drawdown': (np.maximum.accumulate(cumulative_log_returns) - cumulative_log_returns).max() if return_series.count() > 0 else np.nan,
            'sharpe_ratio': (return_series.mean() / return_series.std(ddof=1) * np.sqrt(252)) if return_series.std(ddof=1) > 0 else np.nan,
            'sortino_ratio': (return_series.mean() / return_series[return_series < 0].std(ddof=1) * np.sqrt(252)) if return_series[return_series < 0].std(ddof=1) > 0 else np.nan,
            'CAGR': (np.expm1(cumulative_log_returns.iloc[-1]) + 1) ** (252 / return_series.count()) - 1 if return_series.count() > 0 else np.nan,
            'VaR_5pct': np.percentile(return_series.dropna(), 5) if return_series.count() > 0 else np.nan,
            'count': return_series.count(),
            'mean': return_series.mean(),
            'std_dev': return_series.std(ddof=1),
            'min': return_series.min(),
            'max': return_series.max(),
            'skew': return_series.skew(),
            'kurtosis': return_series.kurtosis(),
        }, name=self.name)
        return stats

    #def _compute_bt_performance(self) -> pd.Series:
    #    bt_performance = self.get_datasets_bt_performance()
    #    if bt_performance.empty:
    ##        return pd.Series(name=self.name, dtype=float)
    #    return bt_performance.mean(axis=0)

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())
    
    def get_datasets_returns(self,dataset_names:list=[]):
        if not dataset_names:
            dataset_names = self.list_datasets()
        returns = pd.DataFrame()
        for dataset_name in dataset_names:
            dataset_return = self.datasets.get(dataset_name).get_returns()
            if dataset_return is not None:
                returns_df = pd.DataFrame()
                returns_df[dataset_name] = dataset_return.copy()
                returns_df = returns_df.reset_index(drop=True)
                returns = pd.concat([returns, returns_df], axis=1)
        return returns

    def get_datasets_bt_performance(self,dataset_names:list=[]):
        if not dataset_names:
            dataset_names = self.list_datasets()
        performance = pd.DataFrame()
        for dataset_name in dataset_names:
            dataset_performance = self.datasets.get(dataset_name).get_bt_performance()
            if dataset_performance is not None and not dataset_performance.empty:
                performance = pd.concat([performance, dataset_performance], axis=0)
        return performance
    
    def get_datasets_performance(self,dataset_names:list=[]):
        if not dataset_names:
            dataset_names = self.list_datasets()
        performance = pd.DataFrame()
        for dataset_name in dataset_names:
            dataset_performance = self.datasets.get(dataset_name).get_performance()
            if dataset_performance is not None and not dataset_performance.empty:
                performance = pd.concat([performance, dataset_performance], axis=0)
        return pd.DataFrame(performance)

    def get_datasets_weights(self,dataset_names:list=[]):
        if not dataset_names:
            dataset_names = self.list_datasets()
        weights = {}
        for dataset_name in dataset_names:
            dataset_weights = self.datasets.get(dataset_name).get_weights()
            if dataset_weights is not None and not dataset_weights.empty:
                weights[dataset_name] = dataset_weights
        return weights

    def get_datasets_asset_values(self, dataset_names: list = [], component: Optional[str] = None):
        """
        Get asset values history for multiple datasets.
        
        Parameters:
        -----------
        dataset_names : list, optional
            List of dataset names to retrieve. If empty, gets all datasets.
        component : str, optional
            Which component to return ('prices', 'positions', 'values', 'portfolio').
            If None, returns all components for each dataset.
        
        Returns:
        --------
        dict
            Dictionary with dataset names as keys and asset values data as values.
            If component is specified, each value is a DataFrame.
            If component is None, each value is a dict of DataFrames.
        
        Examples:
        ---------
        >>> # Get all asset values for all datasets
        >>> all_values = strategy_results.get_datasets_asset_values()
        >>> dataset1_prices = all_values['dataset_1']['prices']
        
        >>> # Get only position data for specific datasets
        >>> positions = strategy_results.get_datasets_asset_values(
        ...     dataset_names=['dataset_1', 'dataset_2'],
        ...     component='positions'
        ... )
        >>> dataset1_positions = positions['dataset_1']
        """
        if not dataset_names:
            dataset_names = self.list_datasets()
        
        asset_values = {}
        for dataset_name in dataset_names:
            dataset_asset_values = self.datasets.get(dataset_name).get_asset_values(component)
            # Include even if empty, for consistency
            asset_values[dataset_name] = dataset_asset_values
        
        return asset_values

    def get_returns(self):
        datasets_returns = self.get_datasets_returns()
        mean_returns = datasets_returns.mean(axis=1)
        mean_returns=pd.DataFrame({'return': mean_returns})
        return mean_returns
    
    def get_performance(self):
        stats = self._compute_performance()

        if stats.empty:
            return pd.DataFrame()
        return stats.to_frame().T

class DatasetResults:

    def __init__(self,name: str,payload: Dict,lookback_periods: int,strategy_name: str):
        self.name: str = name
        self.strategy_name: str = strategy_name
        self.lookback_periods: int = lookback_periods
        self.performance_bt = payload.get('performance', {})
        if 'annual_returns_by_year' in self.performance_bt:
            self.annual_returns = self.performance_bt.get('annual_returns_by_year', {})
            self.performance_bt.pop('annual_returns_by_year', None)
        else:
            self.annual_returns = {}
        
        self.final_value = float(payload.get('final_value', np.nan))
        
        self.returns = self._build_returns_table(payload)
        self.weights_history = self._build_weights_table(payload)
        self.orders = self._build_orders_table(payload)
        self.asset_values_history = self._build_asset_values_table(payload)

    def _build_returns_table(self,payload: Dict) -> pd.DataFrame:
        returns = payload.get('returns', [])
        if not returns:
            return pd.DataFrame(columns=['return'])
        returns_array = np.asarray(returns, dtype=float)

        dates = payload.get('return_dates')
        if dates is not None and len(dates) == len(returns_array):
            index = pd.to_datetime(dates, errors='coerce')
            df = pd.DataFrame({'return': returns_array}, index=index)
            df.index.name = 'date'
            df = df.iloc[self.lookback_periods:]
            return df.sort_index()
        else:
            print(f"Warning: Mismatch in returns and dates length for dataset '{self.name}' in strategy '{self.strategy_name}'. Returning returns without date index.")
        return pd.DataFrame({'return': returns_array})

    def _build_weights_table(self,payload: Dict) -> pd.DataFrame:
        history = payload.get('weights_history', {})
        dates = history.get('rebalancing_dates', [])
        weights_series = history.get('portfolio_weights', [])
        asset_names_series = history.get('asset_names', [])

        if not dates or not weights_series:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for idx, (date, weights) in enumerate(zip(dates, weights_series)):
            if isinstance(asset_names_series, list):
                if len(asset_names_series) == len(weights_series):
                    names = asset_names_series[idx]
                elif asset_names_series:
                    names = asset_names_series[0]
                else:
                    names = None
            else:
                names = None
            if not names:
                names = [f'Asset_{col}' for col in range(len(weights))]
            row = {asset: weight for asset, weight in zip(names, weights)}
            row['date'] = pd.to_datetime(date, errors='coerce')
            rows.append(row)

        weights_df = pd.DataFrame(rows)
        if 'date' in weights_df.columns:
            weights_df = weights_df.set_index('date').sort_index()
        return weights_df

    def _build_orders_table(self,payload: Dict) -> pd.DataFrame:
        order_history = payload.get('order_history', {})
        orders = order_history.get('orders')
        if not orders:
            return pd.DataFrame()
        orders_df = pd.DataFrame(orders)
        orders_df = orders_df.set_index('order_id')
        #print (orders_df)
        #for column in ['created_date', 'executed_date']:
        #    print(column)
        #    if column in orders_df.columns:
        #        print(column)
        #        orders_df[column] = pd.to_datetime(orders_df[column])#, unit='s')
        return orders_df

    def _build_asset_values_table(self, payload: Dict) -> Dict[str, pd.DataFrame]:
        """
        Build DataFrames from asset values history.
        
        Returns a dictionary containing:
        - 'prices': DataFrame of asset prices at each rebalancing
        - 'positions': DataFrame of asset positions (shares) at each rebalancing
        - 'values': DataFrame of asset values (price × shares) at each rebalancing
        - 'portfolio': DataFrame of portfolio-level metrics (total value, cash)
        """
        asset_values_history = payload.get('asset_values_history', {})
        
        dates = asset_values_history.get('rebalancing_dates', [])
        if not dates:
            # Return empty structure if no data
            return {
                'prices': pd.DataFrame(),
                'positions': pd.DataFrame(),
                'values': pd.DataFrame(),
                'portfolio': pd.DataFrame()
            }
        
        # Convert dates to datetime index
        date_index = pd.to_datetime(dates, errors='coerce')
        
        # Get asset names (assuming they're consistent across rebalancing periods)
        asset_names_series = asset_values_history.get('asset_names', [])
        if asset_names_series and len(asset_names_series) > 0:
            asset_names = asset_names_series[0]
        else:
            asset_names = []
        
        # Build prices DataFrame
        prices_data = asset_values_history.get('asset_prices', [])
        df_prices = pd.DataFrame(
            prices_data,
            index=date_index,
            columns=asset_names
        ) if prices_data and asset_names else pd.DataFrame()
        
        # Build positions DataFrame
        positions_data = asset_values_history.get('asset_positions', [])
        df_positions = pd.DataFrame(
            positions_data,
            index=date_index,
            columns=asset_names
        ) if positions_data and asset_names else pd.DataFrame()
        
        # Build values DataFrame
        values_data = asset_values_history.get('asset_values', [])
        df_values = pd.DataFrame(
            values_data,
            index=date_index,
            columns=asset_names
        ) if values_data and asset_names else pd.DataFrame()
        
        # Build portfolio-level DataFrame
        portfolio_values = asset_values_history.get('portfolio_value', [])
        cash_values = asset_values_history.get('cash', [])
        df_portfolio = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'cash': cash_values
        }, index=date_index) if portfolio_values and cash_values else pd.DataFrame()
        
        # Set index name for all DataFrames
        for df in [df_prices, df_positions, df_values, df_portfolio]:
            if not df.empty:
                df.index.name = 'date'
        
        return {
            'prices': df_prices,
            'positions': df_positions,
            'values': df_values,
            'portfolio': df_portfolio
        }

    def _compute_performance(self) -> pd.Series:
        if self.returns.empty:
            return pd.Series(name=self.name, dtype=float)
        return_series = self.returns['return']
        cumulative_log_returns = np.log1p(return_series).cumsum()
        stats = pd.Series({
            'count': return_series.count(),
            'mean': return_series.mean(),
            'std_dev': return_series.std(ddof=1),
            'min': return_series.min(),
            'max': return_series.max(),
            'skew': return_series.skew(),
            'kurtosis': return_series.kurtosis(),
            'annual_return': np.expm1(cumulative_log_returns.iloc[-1]) * (252 / return_series.count()) if return_series.count() > 0 else np.nan,
            'annual_volatility': return_series.std(ddof=1) * np.sqrt(252) if return_series.count() > 1 else np.nan,
            'max_drawdown': (np.maximum.accumulate(cumulative_log_returns) - cumulative_log_returns).max() if return_series.count() > 0 else np.nan,
            'sharpe_ratio': (return_series.mean() / return_series.std(ddof=1) * np.sqrt(252)) if return_series.std(ddof=1) > 0 else np.nan,
            'sortino_ratio': (return_series.mean() / return_series[return_series < 0].std(ddof=1) * np.sqrt(252)) if return_series[return_series < 0].std(ddof=1) > 0 else np.nan,
            'CAGR': (np.expm1(cumulative_log_returns.iloc[-1]) + 1) ** (252 / return_series.count()) - 1 if return_series.count() > 0 else np.nan,
            'VaR_5pct': np.percentile(return_series.dropna(), 5) if return_series.count() > 0 else np.nan,
        }, name=self.name)
        return stats

    def get_performance(self) -> pd.DataFrame:
        stats = self._compute_performance()
        if stats.empty:
            return pd.DataFrame()
        return stats.to_frame().T

    def get_bt_performance(self):
        # Convert performance_bt dict to DataFrame with one row
        if not self.performance_bt:
            return pd.DataFrame()
        df = pd.DataFrame([self.performance_bt], index=[self.name])
        df['final_value'] = self.final_value
        return df
    
    def get_returns(self) -> pd.DataFrame:
        return self.returns.copy()
    
    def get_weights(self) -> pd.DataFrame:
        return self.weights_history.copy()
    
    def get_orders(self) -> pd.DataFrame:
        return self.orders.copy()
    
    def get_asset_values(self, component: Optional[str] = None) -> Dict[str, pd.DataFrame] | pd.DataFrame:
        """
        Get asset values history data.
        
        Parameters:
        -----------
        component : str, optional
            Which component to return. Options:
            - 'prices': Asset prices at each rebalancing
            - 'positions': Asset positions (shares) at each rebalancing
            - 'values': Asset values (price × shares) at each rebalancing
            - 'portfolio': Portfolio-level metrics (total value, cash)
            - None (default): Returns all components as a dictionary
        
        Returns:
        --------
        dict or pd.DataFrame
            If component is None, returns dictionary with all DataFrames.
            If component is specified, returns the requested DataFrame.
        
        Examples:
        ---------
        >>> # Get all asset values data
        >>> all_data = dataset_result.get_asset_values()
        >>> prices_df = all_data['prices']
        >>> positions_df = all_data['positions']
        
        >>> # Get only asset values
        >>> values_df = dataset_result.get_asset_values('values')
        
        >>> # Get portfolio metrics
        >>> portfolio_df = dataset_result.get_asset_values('portfolio')
        """
        if component is None:
            # Return copies of all components
            return {
                key: df.copy() if not df.empty else df 
                for key, df in self.asset_values_history.items()
            }
        
        if component not in self.asset_values_history:
            raise ValueError(
                f"Invalid component '{component}'. "
                f"Valid options are: {list(self.asset_values_history.keys())}"
            )
        
        df = pd.DataFrame(self.asset_values_history[component])
        return df.copy() if not df.empty else df

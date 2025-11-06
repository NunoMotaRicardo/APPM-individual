"""
Validation script for BacktestResults User Guide
Tests basic functionality from the user guide examples
"""

from pathlib import Path
from Backtester.BacktestResults import TestResults

def test_basic_loading():
    """Test basic loading functionality from Quick Start"""
    print("="*80)
    print("TEST 1: Basic Loading (from Quick Start)")
    print("="*80)
    
    try:
        # Load backtest results from a test directory
        test = TestResults("data/selection3/test-4")
        
        # Get aggregated performance statistics for all strategies
        performance_stats = test.get_strategies_bt_performance(aggregator="mean")
        
        print("✓ Successfully loaded test results")
        print(f"✓ Test name: {test.name}")
        print(f"✓ Number of strategies: {len(test.strategies)}")
        print(f"✓ Number of datasets: {len(test.datasets)}")
        print(f"\n✓ Performance stats shape: {performance_stats.shape}")
        print("\nPerformance Statistics (first 5 columns):")
        print(performance_stats.iloc[:, :5].to_string())
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def test_api_methods():
    """Test key API methods"""
    print("\n" + "="*80)
    print("TEST 2: API Methods")
    print("="*80)
    
    try:
        test = TestResults("data/selection3/test-4")
        
        # Test list methods
        strategies = test.list_strategies()
        datasets = test.list_datasets()
        
        print(f"✓ list_strategies(): {strategies}")
        print(f"✓ list_datasets(): Found {len(datasets)} datasets")
        
        # Test get_datasets_info
        info = test.get_datasets_info(dataset_names=[datasets[0]])
        print(f"✓ get_datasets_info(): Retrieved info for {datasets[0]}")
        
        # Test strategy access
        if strategies:
            strategy_name = strategies[0]
            strategy = test.strategies[strategy_name]
            print(f"✓ Accessed strategy: {strategy_name}")
            
            # Test dataset access
            if strategy.list_datasets():
                dataset_name = strategy.list_datasets()[0]
                dataset = strategy.datasets[dataset_name]
                print(f"✓ Accessed dataset: {dataset_name}")
                
                # Test get methods
                returns = dataset.get_returns()
                print(f"✓ get_returns(): Shape {returns.shape}")
                
                performance = dataset.get_performance()
                print(f"✓ get_performance(): Shape {performance.shape}")
                
                weights = dataset.get_weights()
                print(f"✓ get_weights(): Shape {weights.shape}")
                
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_validation():
    """Test the directory validation function from the guide"""
    print("\n" + "="*80)
    print("TEST 3: Directory Validation")
    print("="*80)
    
    test_path = Path("data/selection3/test-4")
    
    # Check required files
    required_files = {
        'test_settings.json': test_path / 'test_settings.json',
        'datasets_info.json': test_path / 'datasets_info.json',
        'datasets folder': test_path / 'datasets',
        'results folder': test_path / 'results',
        'universe_settings.json': test_path.parent / 'universe_settings.json',
        'universe_info.csv': test_path.parent / 'universe_info.csv'
    }
    
    all_found = True
    for name, path in required_files.items():
        if not path.exists():
            print(f"✗ Missing: {name} at {path}")
            all_found = False
        else:
            print(f"✓ Found: {name}")
    
    return all_found

def main():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("BacktestResults User Guide - Validation Tests")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Basic loading
    results.append(("Basic Loading", test_basic_loading()))
    
    # Test 2: API methods
    results.append(("API Methods", test_api_methods()))
    
    # Test 3: Directory validation
    results.append(("Directory Validation", test_directory_validation()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All validation tests passed!")
        print("✓ The BacktestResults module is working correctly.")
        print("✓ You can now use the examples from the User Guide.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

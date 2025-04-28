import unittest
import pandas as pd
import os
import numpy as np
from optimized import load_data, discretize_data, InvalidDataError


class TestOptimizedDiscretization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.create_test_files()

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_test_files()

    @classmethod
    def create_test_files(cls):
        valid_data = pd.DataFrame({
            'attr1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'attr2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'decision': ['A', 'B', 'A', 'B', 'A']
        })
        valid_data.to_csv('valid_data.csv', index=False)

        single_col = pd.DataFrame({'attr1': [1, 2, 3]})
        single_col.to_csv('single_column.csv', index=False)

        open('empty_file.csv', 'a').close()

        all_numeric = pd.DataFrame({
            'attr1': [1, 2, 3],
            'attr2': [4, 5, 6],
            'attr3': [7, 8, 9]
        })
        all_numeric.to_csv('all_numeric.csv', index=False)

        large_data = pd.DataFrame({
            'attr1': np.random.rand(100),
            'attr2': np.random.rand(100),
            'decision': np.random.choice(['A', 'B'], 100)
        })
        large_data.to_csv('large_data.csv', index=False)

    @classmethod
    def cleanup_test_files(cls):
        files = [
            'valid_data.csv', 'single_column.csv', 'empty_file.csv',
            'all_numeric.csv', 'large_data.csv',
            'valid_data.csv_main_discretized.csv',
            'valid_data.csv_secondary_discretized.csv',
            'large_data.csv_main_discretized.csv',
            'large_data.csv_secondary_discretized.csv'
        ]
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def test_load_data_valid(self):
        """Test loading valid data file"""
        data = load_data('valid_data.csv')
        self.assertEqual(len(data.columns), 3)
        self.assertEqual(list(data.columns), ['attr1', 'attr2', 'decision'])

    def test_load_data_nonexistent_file(self):
        """Test loading nonexistent file"""
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')

    def test_load_data_empty_file(self):
        """Test loading empty file"""
        with self.assertRaises(ValueError):
            load_data('empty_file.csv')

    def test_load_data_invalid_structure(self):
        """Test loading data with invalid structure"""
        with self.assertRaises(InvalidDataError):
            load_data('single_column.csv')

        with self.assertRaises(InvalidDataError):
            load_data('all_numeric.csv')

    def test_discretize_data_basic(self):
        """Test basic discretization functionality"""
        data = pd.DataFrame({
            'attr1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'attr2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'decision': ['A', 'B', 'A', 'B', 'A']
        })

        discretized, stats = discretize_data(data.copy(), verbose=False)

        self.assertEqual(len(discretized.columns), 3)
        self.assertEqual(list(discretized.columns), ['attr1', 'attr2', 'decision'])
        self.assertTrue(all(';' in str(val) for val in discretized['attr1'].values))
        self.assertTrue(all(';' in str(val) for val in discretized['attr2'].values))
        self.assertEqual(list(discretized['decision']), ['A', 'B', 'A', 'B', 'A'])
        self.assertGreater(stats['separated_pairs'], 0)
        self.assertGreater(stats['cuts_added'], 0)
        self.assertGreater(stats['coverage'], 0)

    def test_discretize_criteria_comparison(self):
        """Test that secondary criterion produces fewer cuts"""
        data = pd.DataFrame({
            'attr1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'attr2': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            'decision': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

        _, main_stats = discretize_data(data.copy(), use_secondary_criterion=False, verbose=False)
        _, secondary_stats = discretize_data(data.copy(), use_secondary_criterion=True, verbose=False)

        self.assertLessEqual(secondary_stats['cuts_added'], main_stats['cuts_added'])
        self.assertGreaterEqual(main_stats['coverage'], secondary_stats['coverage'])

    def test_discretize_performance(self):
        """Test performance with larger dataset"""
        data = pd.read_csv('large_data.csv')

        import time
        start_time = time.time()
        discretized, stats = discretize_data(data.copy(), verbose=False)
        elapsed = time.time() - start_time

        print(f"\noptimized.py took {elapsed:.2f} seconds")
        self.assertEqual(len(discretized), len(data))
        self.assertGreater(stats['separated_pairs'], 0)
        self.assertGreater(stats['cuts_added'], 0)

    def test_already_separated_data(self):
        """Test data that's already perfectly separated"""
        data = pd.DataFrame({
            'attr1': [1.0, 2.0, 1.1, 2.1],
            'decision': ['A', 'B', 'A', 'B']
        })

        discretized, stats = discretize_data(data.copy(), verbose=False)
        self.assertEqual(stats['coverage'], 1.0)
        self.assertEqual(stats['cuts_added'], 1)
        unique_intervals = discretized['attr1'].unique()
        self.assertEqual(len(unique_intervals), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
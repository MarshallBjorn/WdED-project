import pandas as pd
import numpy as np
from itertools import combinations
import os


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses data for discretization
    with comprehensive error handling

    Args:
        file_path: path to the '.csv' data file

    Returns:
        DataFrame with preprocessed data
        and list of numeric columns for discretization

    Raises:
        FileNotFoundError: if the specified file doesn't exist
        ValueError: if the file is empty or contains no numeric columns
        pd.errors.EmptyDataError: if the file contains no data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exists.")

    # Check if file is not empty
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Error: The file '{file_path}' is empty.")

    # Try to read the file
    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: The file '{file_path}' contains no data.")

    # Identify numeric columns for discretization
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    print(f"Numeric columns identified for discretization: {list(numeric_cols)}")

    # Sprawdzenie brakujących wartości - tylko w kolumnach numerycznych
    if data[numeric_cols].isnull().any().any():
        print("Note: Missing values found in numeric columns. Filling with medians.")
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Handle missing values in non-numeric columns - fill with the most frequent value
    non_numeric_cols = data.select_dtypes(exclude=["float64", "int64"]).columns
    for col in non_numeric_cols:
        if data[col].isnull().any():
            most_frequent = data[col].mode()[0]
            print(
                f"Note: Missing values in column '{col}'."
                f"Filling with most frequent value '{most_frequent}'."
            )
            data[col] = data[col].fillna(most_frequent)

    return data, numeric_cols


if __name__ == "__main__":
    try:
        print(load_and_preprocess_data("qewrty.csv"))
    except FileNotFoundError as e:
        print(e, "\n")
    finally:
        print(load_and_preprocess_data("test_data.csv"), "\n")
        print(load_and_preprocess_data("iris.csv"), "\n")

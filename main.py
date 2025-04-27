import pandas as pd
import os


class InvalidDataError(Exception):
    """Raised when the input data structure is invalid for discretization."""
    pass


def load_data(file_path):
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
        raise FileNotFoundError(
            f"Error: The file '{file_path}' does not exists.")

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
    print(f"Numeric columns identified for discretization: {
          list(numeric_cols)}")

    if len(data.columns) < 2:
        raise InvalidDataError(
            """Error: The dataset must contain at least one attribute and one decision column.""")

    if len(numeric_cols) != len(data.columns) - 1:
        raise InvalidDataError(
            """Error: Invalid dataset. Expected n-1 numerical columns and a decision one (nth).""")

    return data


def prepare_for_discretization(data):
    decision_column = data.columns[-1]
    attribute_columns = data.columns[:-1].tolist()

    return attribute_columns, decision_column


def discretize_data(data):
    attributes, decision = prepare_for_discretization(data)

    cuts = {attr: [] for attr in attributes}

    # Todo:
    #   generowanie par
    #   implementacja kryteriów
    #   wybieranie najlepszych cięć

    # Discretize attributes based on cuts
    discretized = []
    for idx, row in data.iterrows():
        new_row = []
        for attr in attributes:
            value = row[attr]
            attr_cuts = cuts[attr]

            if not attr_cuts:
                interval = "(-inf; inf)"
            else:
                left = "-inf"
                right = "inf"
                for cut in attr_cuts:
                    if value <= cut:
                        right = cut
                        break
                    left = cut
                interval = f"({left}; {right}]"

            new_row.append(interval)

        new_row.append(row[decision])
        discretized.append(new_row)

    return pd.DataFrame(discretized)


if __name__ == "__main__":
    test_files = ["qewrty.csv", "test_data.csv", "iris.csv"]

    for file in test_files:
        try:
            print(f"Processing file: {file}")
            data = load_data(file)
            attributes, decision = prepare_for_discretization(data)
            print(data.head(), "\n")
        except (FileNotFoundError, ValueError, InvalidDataError) as e:
            print(e, "\n")

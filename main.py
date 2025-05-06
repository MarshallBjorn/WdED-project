import itertools
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
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

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

    # if len(data.columns) < 2:
    #     raise InvalidDataError(
    #         "Error: The dataset must contain at least one attribute and one decision column."
    #     )

    # if len(numeric_cols) != len(data.columns) - 1:
    #     raise InvalidDataError(
    #         "Error: Invalid dataset. Expected n-1 numerical columns and a decision one (nth)."
    #     )

    return data


def prepare_for_discretization(data):
    decision_column = data.columns[-1]
    attribute_columns = data.columns[:-1].tolist()
    return attribute_columns, decision_column


def generate_object_pairs(data, decision_col):
    """Generates pairs of indexes where decision attribute differs."""
    return [
        (i, j)
        for (i, row_i), (j, row_j) in itertools.combinations(data.iterrows(), 2)
        if row_i[decision_col] != row_j[decision_col]
    ]


def find_possible_cuts(values):
    """Finds possible cuts for given sorted values."""
    unique_values = sorted(set(values))
    return [
        (unique_values[i] + unique_values[i + 1]) / 2
        for i in range(len(unique_values) - 1)
    ]


def check_separation(row1, row2, cuts):
    """Checks if two rows are separated by given cuts."""
    for cut in cuts:
        if (row1 <= cut < row2) or (row2 <= cut < row1):
            return True
    return False


def discretize_data(data, use_secondary_criterion=False, verbose=True):
    """
    Discretizes data using either the main criterion (maximize separated pairs)
    or secondary criterion (minimize number of intervals)

    Args:
        data: input DataFrame
        use_secondary_criterion: if True, uses the secondary criterion

    Returns:
        discretized DataFrame and statistics about the discretization
    """
    attributes, decision = prepare_for_discretization(data)
    object_pairs = generate_object_pairs(data, decision)
    (
        print(f"Generated {len(object_pairs)} object pairs with different decisions.")
        if verbose
        else None
    )

    cuts = {attr: [] for attr in attributes}
    separated_pairs = set()

    stats = {
        "total_pairs": len(object_pairs),
        "separated_pairs": 0,
        "cuts_added": 0,
        "cuts_per_attribute": {attr: 0 for attr in attributes},
    }

    while True:
        best = {
            "gain": 0,
            "attr": None,
            "cut": None,
            "new_separations": set(),
            "cuts_count": float("inf"),
        }

        # Try every attribute and every possible cut
        for attr in attributes:
            possible_cuts = find_possible_cuts(data[attr])

            for cut in possible_cuts:
                temp_cuts = sorted(cuts[attr] + [cut])
                new_separations = {
                    (i, j)
                    for i, j in object_pairs
                    if (i, j) not in separated_pairs
                    and check_separation(data.at[i, attr], data.at[j, attr], temp_cuts)
                }
                gain = len(new_separations)
                cuts_count = len(temp_cuts)

                if use_secondary_criterion:
                    if gain > 0 and (
                        cuts_count < best["cuts_count"]
                        or (cuts_count == best["cuts_count"] and gain > best["gain"])
                    ):
                        best.update(
                            {
                                "gain": gain,
                                "attr": attr,
                                "cut": cut,
                                "new_separations": new_separations,
                                "cuts_count": cuts_count,
                            }
                        )
                else:
                    if gain > best["gain"]:
                        best.update(
                            {
                                "gain": gain,
                                "attr": attr,
                                "cut": cut,
                                "new_separations": new_separations,
                                "cuts_count": cuts_count,
                            }
                        )

        if best["gain"] == 0:
            break

        # Apply the best cut
        cuts[best["attr"]].append(best["cut"])
        cuts[best["attr"]] = sorted(cuts[best["attr"]])
        separated_pairs.update(best["new_separations"])

        stats["cuts_added"] += 1
        stats["cuts_per_attribute"][best["attr"]] += 1
        stats["separated_pairs"] = len(separated_pairs)

        if verbose:
            print(
                f"Added cut {best['cut']} on attribute '{best['attr']}', separated {best['gain']} new pairs."
            )
            print(f"Total separated pairs: {len(separated_pairs)}/{len(object_pairs)}")
            print(f"Current cuts: {cuts}\n")

    # Discretize the dataset
    discretized_rows = []
    for idx, row in data.iterrows():
        new_row = []
        for attr in attributes:
            value = row[attr]
            attr_cuts = cuts[attr]
            if not attr_cuts:
                interval = "(-inf; inf)"
            else:
                left = "-inf"
                for cut in attr_cuts:
                    if value <= cut:
                        interval = f"({left}; {cut}]"
                        break
                    left = cut
                else:
                    interval = f"({left}; inf)"
            new_row.append(interval)
        new_row.append(row[decision])
        discretized_rows.append(new_row)

    discretized_df = pd.DataFrame(discretized_rows, columns=attributes + [decision])

    stats["coverage"] = (
        stats["separated_pairs"] / stats["total_pairs"] if stats["total_pairs"] else 0
    )
    stats["average_cuts_per_attribute"] = (
        stats["cuts_added"] / len(attributes) if attributes else 0
    )

    return discretized_df, stats


def compare_criteria(data):
    """Compares the main and secondary criteria"""
    print("\n=== Using MAIN CRITERION (maximize separated pairs) ===")
    main_result, main_stats = discretize_data(data, use_secondary_criterion=False)

    print("\n=== Using SECONDARY CRITERION (minimize number of intervals) ===")
    secondary_result, secondary_stats = discretize_data(
        data, use_secondary_criterion=True
    )

    print("\n=== Comparison Results ===")
    print(f"{'Metric':<30} {'Main Criterion':<20} {'Secondary Criterion':<20}")
    print(
        f"{'Number of separated pairs':<30} {main_stats['separated_pairs']:<20} {secondary_stats['separated_pairs']:<20}"
    )
    print(
        f"{'Total cuts added':<30} {main_stats['cuts_added']:<20} {secondary_stats['cuts_added']:<20}"
    )
    print(
        f"{'Coverage (separated/total)':<30} {main_stats['coverage']:<20.2%} {secondary_stats['coverage']:<20.2%}"
    )

    return main_result, secondary_result


def save_discretized_data(discretized_df, output_path):
    """Saves the discretized DataFrame to a CSV file."""
    discretized_df.to_csv(output_path, index=False)
    print(f"Discretized data saved to '{output_path}'")


if __name__ == "__main__":
    test_files = ["data2.csv", "data3.csv"]

    for file in test_files:
        print(f"\n{'='*50}")
        print(f"Processing file: {file}")

        try:
            data = load_data(file)
            attributes, decision = prepare_for_discretization(data)
            print(data.head(), "\n")

            main_disc, secondary_disc = compare_criteria(data)

            print("\nMain criterion discretization:")
            print(main_disc.head())

            print("\nSecondary criterion discretization:")
            print(secondary_disc.head())

            save_discretized_data(main_disc, f"{file}_main_discretized.csv")
            save_discretized_data(secondary_disc, f"{file}_secondary_discretized.csv")

        except (FileNotFoundError, ValueError, InvalidDataError) as e:
            print(e, "\n")

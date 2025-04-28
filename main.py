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
    print(f"Numeric columns identified for discretization: {list(numeric_cols)}")

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


def discretize_data(data, use_secondary_criterion=False):
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

    cuts = {attr: [] for attr in attributes}

    # Generate all object pairs with different decisions
    object_pairs = []
    for (idx1, row1), (idx2, row2) in itertools.combinations(data.iterrows(), 2):
        if row1.iloc[-1] != row2.iloc[-1]:  # different decision
            object_pairs.append((idx1, idx2))

    print(f"Generated {len(object_pairs)} object pairs with different decisions.")

    # Keep track of which pairs are separated
    separated_pairs = set()
    
    # Statistics
    stats = {
        'total_pairs': len(object_pairs),
        'separated_pairs': 0,
        'cuts_added': 0,
        'cuts_per_attribute': {attr: 0 for attr in attributes}
    }

    while True:
        best_attr = None
        best_cut = None
        best_gain = 0
        best_new_separations = set()
        best_cuts_count = float('inf')  # for secondary criterion

        # Try every attribute and every possible cut
        for attr in attributes:
            values = data[attr].unique()
            possible_cuts = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            possible_cuts = sorted(set(possible_cuts))  # unique cuts

            for cut in possible_cuts:
                temp_cuts = cuts[attr] + [cut]
                temp_cuts = sorted(temp_cuts)

                new_separations = set()

                for idx1, idx2 in object_pairs:
                    row1 = data.loc[idx1]
                    row2 = data.loc[idx2]

                    # Check if the two objects are separated by this set of cuts
                    for c in temp_cuts:
                        if (row1[attr] <= c and row2[attr] > c) or (row2[attr] <= c and row1[attr] > c):
                            new_separations.add((idx1, idx2))
                            break

                gain = len(new_separations - separated_pairs)
                cuts_count = len(temp_cuts)
                
                # Evaluate based on selected criterion
                if use_secondary_criterion:
                    # Secondary criterion: minimize number of cuts while separating at least one new pair
                    if gain > 0 and (cuts_count < best_cuts_count or 
                                    (cuts_count == best_cuts_count and gain > best_gain)):
                        best_gain = gain
                        best_attr = attr
                        best_cut = cut
                        best_new_separations = new_separations
                        best_cuts_count = cuts_count
                else:
                    # Main criterion: maximize number of separated pairs
                    if gain > best_gain:
                        best_gain = gain
                        best_attr = attr
                        best_cut = cut
                        best_new_separations = new_separations
                        best_cuts_count = cuts_count

        if best_gain == 0:
            # No further improvement
            break

        # Apply the best cut
        cuts[best_attr].append(best_cut)
        cuts[best_attr] = sorted(cuts[best_attr])
        separated_pairs.update(best_new_separations)
        stats['cuts_added'] += 1
        stats['cuts_per_attribute'][best_attr] += 1
        stats['separated_pairs'] = len(separated_pairs)

        print(f"Added cut {best_cut} on attribute '{best_attr}', separated {best_gain} new pairs.")
        print(f"Total separated pairs: {len(separated_pairs)}/{len(object_pairs)}")
        print(f"Current cuts: {cuts}\n")

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

    discretized_df = pd.DataFrame(discretized, columns=attributes + [decision])
    
    # Calculate statistics
    stats['coverage'] = stats['separated_pairs'] / stats['total_pairs'] if stats['total_pairs'] > 0 else 0
    stats['average_cuts_per_attribute'] = stats['cuts_added'] / len(attributes) if len(attributes) > 0 else 0
    
    return discretized_df, stats


def compare_criteria(data):
    """Compares the main and secondary criteria"""
    print("\n=== Using MAIN CRITERION (maximize separated pairs) ===")
    main_result, main_stats = discretize_data(data, use_secondary_criterion=False)
    
    print("\n=== Using SECONDARY CRITERION (minimize number of intervals) ===")
    secondary_result, secondary_stats = discretize_data(data, use_secondary_criterion=True)
    
    print("\n=== Comparison Results ===")
    print(f"{'Metric':<30} {'Main Criterion':<20} {'Secondary Criterion':<20}")
    print(f"{'Number of separated pairs':<30} {main_stats['separated_pairs']:<20} {secondary_stats['separated_pairs']:<20}")
    print(f"{'Total cuts added':<30} {main_stats['cuts_added']:<20} {secondary_stats['cuts_added']:<20}")
    print(f"{'Coverage (separated/total)':<30} {main_stats['coverage']:<20.2%} {secondary_stats['coverage']:<20.2%}")
    
    return main_result, secondary_result


if __name__ == "__main__":
    test_files = ["qewrty.csv", "test_data.csv", "iris.csv"]

    for file in test_files:
        try:
            print(f"\n{'='*50}")
            print(f"Processing file: {file}")
            data = load_data(file)
            attributes, decision = prepare_for_discretization(data)
            print(data.head(), "\n")
            
            # Compare both criteria
            main_disc, secondary_disc = compare_criteria(data)
            
            print("\nMain criterion discretization:")
            print(main_disc.head())
            
            print("\nSecondary criterion discretization:")
            print(secondary_disc.head())
            
        except (FileNotFoundError, ValueError, InvalidDataError) as e:
            print(e, "\n")
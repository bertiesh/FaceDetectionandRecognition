import os
from collections import defaultdict

def count_matches(database_dir, queries_dir):
    # Extract base names (without numbering) from filenames
    def get_base_name(filename):
        return "_".join(filename.split("_")[:-1])  # Remove last part after last underscore
    
    # Get all base names from the database
    database_names = defaultdict(int)
    for filename in os.listdir(database_dir):
        base_name = get_base_name(filename)
        database_names[base_name] += 1
    
    # Check matches in queries
    match_counts = defaultdict(int)
    for filename in os.listdir(queries_dir):
        base_name = get_base_name(filename)
        if base_name in database_names:
            match_counts[base_name] += 1
    
    # Count statistics
    total_matches = len(match_counts)
    at_least_2_matches = sum(1 for v in match_counts.values() if v >= 2)
    at_least_5_matches = sum(1 for v in match_counts.values() if v >= 5)
    
    print(f"Total queries with at least one match: {total_matches}")
    print(f"Total queries with at least 2 matches: {at_least_2_matches}")
    print(f"Total queries with at least 5 matches: {at_least_5_matches}")

# Example usage
count_matches('sample_database', 'sample_queries')

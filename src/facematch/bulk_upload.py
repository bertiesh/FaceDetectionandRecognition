import csv
import os
import pandas as pd

def upload_embedding_to_database(data, database_filepath):
    csv_file = database_filepath
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode="w", newline="") as file:
        # # Create a CSV writer object
        # writer = csv.DictWriter(file, fieldnames=data[0].keys())

        # # Write the header (column names)
        # writer.writeheader()

        # # Write each dictionary as a row in the CSV file
        # writer.writerows(data)

        df = pd.DataFrame(data)
        df["embedding"] = df["embedding"].apply(lambda x: ",".join(map(str, x)))
        df["bbox"] = df["bbox"].apply(lambda x: ",".join(map(str, x)))
        df.to_csv(csv_file, index=False)


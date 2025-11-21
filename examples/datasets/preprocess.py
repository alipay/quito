import pandas as pd
from pathlib import Path
import glob

def convert_csv_to_parquet(input_dir, output_dir):
    """
    Convert all CSV files in the input directory to Parquet format in the output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(list(input_path.glob("*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to convert.")

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Define output filename
            output_file = output_path / f"{csv_file.stem}.parquet"
            
            # Save as Parquet
            # Using 'pyarrow' engine for better compatibility with Hugging Face Datasets
            df.to_parquet(output_file, index=False, engine='pyarrow')
            print(f"Saved to {output_file}")
            
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")

if __name__ == "__main__":
    # Default paths assume the script is run from its location or similar structure
    # Adjust these paths as needed or use argparse for CLI usage
    current_dir = Path(__file__).parent
    input_directory = current_dir / "raw_csvs" / "open_hour_train"
    output_directory = current_dir / "parquet_data" / "open_hour_train"
    
    print(f"Converting files from {input_directory} to {output_directory}")
    convert_csv_to_parquet(input_directory, output_directory)


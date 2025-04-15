import zipfile
import json
import os
import pandas as pd
from pathlib import Path

def load_jsonl_to_dataframe(zip_path='data/test.zip', jsonl_filename='text.jsonl'):
    """
    Unzips the specified JSONL file from a zip archive and returns its contents as a pandas DataFrame.
    
    Args:
        zip_path (str): Path to the zip file relative to the current directory
        jsonl_filename (str): Name of the JSONL file to extract from the zip
        
    Returns:
        pandas.DataFrame: DataFrame containing the JSONL data
    """
    try:
        # Get the absolute path to the zip file
        zip_file_path = Path(zip_path).resolve()
        
        print(f"Looking for zip file at: {zip_file_path}")
            
        # Open and extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # List all files in the zip
            file_list = zip_ref.namelist()
            print(f"Files in zip: {file_list}")
            
            # Find the JSONL file
            jsonl_file = next((f for f in file_list if f.endswith('.jsonl')), None)
                
            print(f"Found JSONL file: {jsonl_file}")
            
            # Extract and read the JSONL file
            data_list = []
            with zip_ref.open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Decode bytes to string and parse JSON
                        # Use 'replace' to handle any invalid characters
                        json_obj = json.loads(line.decode('utf-8', errors='replace'))
                        data_list.append(json_obj)
                        
                        if line_num % 1000 == 0:  # Progress update every 1000 lines
                            print(f"Processed {line_num} lines...")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error on line {line_num}: {str(e)}")
                
        print(f"Successfully loaded {len(data_list)} records from {zip_path}")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data_list)
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the dataset
    medical_df = load_jsonl_to_dataframe()
    
    # Display basic information
    
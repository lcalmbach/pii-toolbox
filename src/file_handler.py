import pandas as pd
import os
from pathlib import Path
import mimetypes

def detect_csv_separator(file_path, encoding='utf-8', sample_size=1024):
    """
    Detect the separator character in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding (default: 'utf-8')
        sample_size (int): Number of bytes to read for detection (default: 1024)
    
    Returns:
        str: Detected separator character (',', ';', or '\t')
    """
    allowed_separators = [',', ';', '\t']
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample = file.read(sample_size)
        
        # Count occurrences of each separator in the sample
        separator_counts = {}
        for sep in allowed_separators:
            separator_counts[sep] = sample.count(sep)
        
        # Return the separator with the highest count
        detected_separator = max(separator_counts, key=separator_counts.get)
        
        # If no separators found or all have zero count, default to comma
        if separator_counts[detected_separator] == 0:
            return ','
            
        return detected_separator
        
    except Exception as e:
        print(f"Error detecting separator: {e}")
        return ','  # Default to comma


def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using common encodings.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: Detected encoding
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file.read(1024)  # Try to read a sample
            return encoding
        except UnicodeDecodeError:
            continue
    
    # If all fail, return utf-8 as default
    return 'utf-8'


def open_file(file_path):
    """
    Open a file based on its extension (.xlsx or .csv) and return a pandas DataFrame.
    For CSV files, automatically detect the separator character.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        pandas.DataFrame: The loaded data
        
    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If file doesn't exist
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.xlsx':
            # Open Excel file
            df = pd.read_excel(file_path)
            print(f"Successfully opened Excel file: {file_path}")
            return df
            
        elif file_extension == '.csv':
            # Detect encoding first
            encoding = detect_file_encoding(file_path)
            
            # Detect separator
            separator = detect_csv_separator(file_path, encoding=encoding)
            
            # Open CSV file with detected separator
            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            print(f"Successfully opened CSV file: {file_path}")
            print(f"Detected separator: '{separator}' ({'tab' if separator == '\\t' else separator})")
            print(f"Detected encoding: {encoding}")
            return df
            
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Only .xlsx and .csv files are supported.")
            
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        raise


def get_file_info(file_path):
    """
    Get information about a file including extension, size, and detected properties.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        dict: File information
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    file_info = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "extension": Path(file_path).suffix.lower(),
        "size_bytes": os.path.getsize(file_path)
    }
    
    # Add CSV-specific information
    if file_info["extension"] == '.csv':
        try:
            encoding = detect_file_encoding(file_path)
            separator = detect_csv_separator(file_path, encoding=encoding)
            file_info.update({
                "encoding": encoding,
                "separator": separator,
                "separator_name": "tab" if separator == '\t' else separator
            })
        except Exception as e:
            file_info["detection_error"] = str(e)
    
    return file_info


# Example usage and testing
if __name__ == "__main__":
    # Test with the existing CSV file in the project
    test_file = "herzfrequenz.csv"
    
    if os.path.exists(test_file):
        print("=== File Information ===")
        info = get_file_info(test_file)
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("\n=== Opening File ===")
        try:
            df = open_file(test_file)
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test file {test_file} not found")


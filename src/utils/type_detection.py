import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, Any

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect column types for a DataFrame.
    
    Valid types: string, integer, float, boolean, date
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary mapping column names to detected types
    """
    column_types = {}
    
    for column in df.columns:
        series = df[column].dropna()  # Remove NaN values for analysis
        
        if len(series) == 0:
            column_types[column] = 'string'  # Default for empty columns
            continue
            
        detected_type = _detect_series_type(series)
        column_types[column] = detected_type
    
    return column_types

def _detect_series_type(series: pd.Series) -> str:
    """Detect the type of a pandas Series."""
    
    # Check for boolean first (most specific)
    if _is_boolean(series):
        return 'boolean'
    
    # Check for date
    if _is_date(series):
        return 'date'
    
    # Check for integer
    if _is_integer(series):
        return 'integer'
    
    # Check for float
    if _is_float(series):
        return 'float'
    
    # Default to string
    return 'string'

def _is_boolean(series: pd.Series) -> bool:
    """Check if series contains boolean values."""
    if series.dtype == bool:
        return True
    
    # Check for string representations of booleans
    unique_values = set(str(val).lower().strip() for val in series.unique())
    boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
    
    return unique_values.issubset(boolean_values) and len(unique_values) <= 2

def _is_date(series: pd.Series) -> bool:
    """Check if series contains date values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # Try to parse as dates
    sample_size = min(100, len(series))
    sample = series.head(sample_size)
    
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
    ]
    
    for value in sample:
        str_value = str(value).strip()
        if any(re.match(pattern, str_value) for pattern in date_patterns):
            try:
                pd.to_datetime(str_value)
                continue
            except:
                return False
        else:
            return False
    
    return len(sample) > 0

def _is_integer(series: pd.Series) -> bool:
    """Check if series contains integer values."""
    if pd.api.types.is_integer_dtype(series):
        return True
    
    # Check if all values can be converted to integers
    try:
        for value in series:
            if pd.isna(value):
                continue
            float_val = float(value)
            if not float_val.is_integer():
                return False
        return True
    except (ValueError, TypeError):
        return False

def _is_float(series: pd.Series) -> bool:
    """Check if series contains float values."""
    if pd.api.types.is_float_dtype(series):
        return True
    
    # Check if all values can be converted to floats
    try:
        for value in series:
            if pd.isna(value):
                continue
            float(value)
        return True
    except (ValueError, TypeError):
        return False

def print_column_types(df: pd.DataFrame) -> None:
    """Print detected column types in a formatted way."""
    types = detect_column_types(df)
    
    print("Detected Column Types:")
    print("-" * 40)
    for column, dtype in types.items():
        print(f"{column:<25} | {dtype}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample DataFrame for testing
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000.5, 60000.0, 75000.25],
        'is_active': [True, False, True],
        'join_date': ['2023-01-15', '2022-06-20', '2021-03-10'],
        'score': ['85', '92', '78'],  # String numbers
        'status': ['yes', 'no', 'yes']  # String booleans
    }
    
    df = pd.DataFrame(sample_data)
    
    # Detect and print types
    print_column_types(df)
    
    # Get types as dictionary
    column_types = detect_column_types(df)
    print(f"\nColumn types dictionary: {column_types}")

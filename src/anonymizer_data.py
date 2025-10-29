import streamlit as st
import json
import pandas as pd
import random
from faker import Faker
from pathlib import Path
from datetime import datetime, timedelta
import unicodedata
import re
import hashlib
from io import BytesIO
import numpy as np
from enum import Enum
from file_handler import open_file
from typing import Dict, Any
from utils.type_detection import detect_column_types

fake = Faker("de_DE")
url_addresses = "https://data.bs.ch/api/explore/v2.1/catalog/datasets/100259/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=str_name,hausnr,hausnr_zus,plz,ort"
url_first_names = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/100129/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=vorname,geschlecht,anzahl&where=jahr={datetime.now().year-2}"
url_last_names = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/100127/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=nachname,anzahl&where=jahr={datetime.now().year-2}"
address_file = "./src/data/100259.parquet"
first_name_file = "./src/data/100129.parquet"
last_name_file = "./src/data/100127.parquet"
plz_file = "./src/data/plz.csv"

OGDBS = 'ogd-bs'

class genderEnum(Enum):
    MALE = "M"
    FEMALE = "F"

def is_valid_canton(cantons: list, canton_short):
    return canton_short.upper() in cantons
    

def get_ogd_data(file: str, url: str):
    """
    Get the addresses from a CSV file.

    :return: DataFrame with addresses
    """
    if Path(file).exists():
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(url, sep=";")
        if "vorname" in df.columns:
            df = df[df["vorname"].str.lower() != "übrige"]
            expanded_df = pd.DataFrame(
                df.apply(
                    lambda row: [[row["vorname"], row["geschlecht"]]]
                    * row["anzahl"],
                    axis=1,
                )
                .explode()
                .tolist(),
                columns=["vorname", "geschlecht"],
            )
            df = expanded_df
        elif "nachname" in df.columns:
            df = df[df["nachname"].str.lower() != "übrige"]
            expanded_df = pd.DataFrame(
                df.apply(lambda row: [row["nachname"]] * row["anzahl"], axis=1)
                .explode()
                .tolist(),
                columns=["nachname"],
            )
            df = expanded_df
        df.to_parquet(file)
    return df


def get_faker_function(column_name, column_type):
    if column_name.lower() in ("name", "nachname", "nach_name", "last_name"):
        return {
            "function": "last_name",
            "source": "ogd-bs",
            "percent_null": 0
        }
    elif column_name.lower() in ("vorname", "vor_name", "first_name"):
        return {
            "function": "first_name",
            "source": "ogd-bs",
            "percent_null": 0
        }
    elif column_name.lower() in ("strasse"):
        return {
            "function": "street_name",
            "source": "ogd-bs",
            "location_field": "<location field>",
            "percent_null": 0
        }     
    elif column_name.lower() in ("hnr", "hausnummer"):
        return {
            "function": "house_number",
            "min": 1,
            "max": 100,
            "pct_with_suffix": 2,
            "suffixes": ["a", "b", "c", "d"]
        }   
    elif column_type == "integer":
        return {
            "function": "random_int",
            "min": 100,
            "max": 200
        }  
    elif column_type == "float":
        return {
            "function": "float_random",
            "digits": 2,
            "min": 100,
            "max": 200
        }  
    elif column_type == "boolean":
        return {
            "function": "boolean_random",
            "true": 'j',
            "false": 'n'
        }  
    else:
        return {"function": None}
    
        
def generate_json_config(file_name: str):
    """address =
    Generate a JSON template basemagicd on the columns of a DataFrame.
    Each column is given a default configuration for anonymization.

    :param df: pandas DataFrame
    :return: Dictionary template for JSON configuration
    """
    df = open_file(file_name)
    file_path = Path(file_name)
    extension = file_path.suffix.lower()
    supported_extensions = ['.xlsx', '.csv']
    if not extension in supported_extensions:
        st.warning(f"Unsupported file type: {extension}")
    
    new_file_path = file_path.with_stem(file_path.stem + "_anonymized")
    json_file_path = file_path.with_suffix('.json')
    # file_base = file_path.stem 

    column_types = detect_column_types(df) 

    config = {
        "filename": str(new_file_path),
        "format": extension,
        "separator": ";",
        "encoding": "utf-8",
        "rows": 1000 if len(df) == 0 else len(df),
        "columns": {}
    }
    
    for column in df.columns:
        # Default configuration
        entry = {
            "data_type": column_types[column]
        }
        for key, value in get_faker_function(column, column_types[column]).items():
            entry[key] = value
        
        config['columns'][column] = entry
    st.write(config)
    with open(json_file_path, "w", encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    return json_file_path

    
class DataGenerator():
    def __init__(self, config):
        """Parst die übergebene JSON-Konfiguration und speichert relevante Werte."""
        self.config = config

        # Werte aus JSON zuweisen
        self.filename = self.config.get("filename", "output.csv")
        self.format = self.config.get("format", "csv")
        self.separator = self.config.get("separator", ",")
        self.encoding = self.config.get("encoding", "utf-8")
        self.rows = self.config.get("rows", 1000)
        self.columns = self.config.get("columns", {})

        self.addresses = get_ogd_data(url=url_addresses, file=address_file)
        self.streets = self.addresses[['str_name', 'ort', 'plz']].drop_duplicates()
        self.first_names = get_ogd_data(url=url_first_names, file=first_name_file)
        # map female sex from W to F
        self.first_names['geschlecht'] = self.first_names['geschlecht'].replace({'W': 'F'})     
        self.last_names = get_ogd_data(url=url_last_names, file=last_name_file)
        self.plz = pd.read_csv(plz_file, sep='\t')
        self.cantons = list(self.plz['kanton_name_kurz'].unique())

    def remove_values(self, arr: np.array, pct: float):
        if pct > 0:
            num_missing = int(pct / 100 * self.rows)
            missing_indices = random.sample(range(self.rows), num_missing)
            arr = arr.astype(object)
            for idx in missing_indices:
                arr[idx] = None
        elif pct >=100:
            arr = np.array(self.rows, None, dtype=object)
        return arr

    def incremental_int(self, config:dict, data: pd.DataFrame):
        arr = np.array([])
        if config['sequential']:
            arr = np.arange(config['min_value'], config['min_value'] + self.rows)
        else:
            arr = np.random.randint(config['min_value'], config['max_value'], self.rows)
        
        if config['percent_null'] > 0:
            num_missing = int(0.1 * self.rows)
            missing_indices = random.sample(range(self.rows), num_missing)
            arr = arr.astype(object)
            for idx in missing_indices:
                arr[idx] = None
        return arr

    def random_int(self, config:dict, data: pd.DataFrame):
        # filsl arr with ranom int values
        arr = np.random.randint(config['min'], config['max'], self.rows)
        
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def random_float(self, config:dict, data: pd.DataFrame):
        # filsl arr with ranom int values
        arr = np.random.uniform(config['min'], config['max'], self.rows)
        
        if 'digits' in config:
            arr = np.round(arr, config["digits"])
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def normal_float(self, config:dict, data: pd.DataFrame):
        # fills a 
        arr = np.random.normal(loc=config["mean"], scale=config["std"], size=self.rows)
        if "min" in config:
            arr =np.clip(arr, config["min"], np.inf) 
        if "max" in config:
             arr = np.minimum(arr, config["max"])
        arr = np.round(arr, config["digits"])
        
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def normal_int(self, config:dict, data: pd.DataFrame):
        # fills a 
        arr = np.random.normal(loc=config["mean"], scale=config["std"], size=self.rows)
        if "min" in config:
            arr = np.clip(arr, np.inf, config["min"])
        if "max" in config:
             arr = np.clip(arr, -np.inf, config["max"])

        arr = np.round(arr).astype(int)
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def integer_grouped(self, config: dict, data: pd.DataFrame):
        arr = list(range(config['values'][0], config['values'][1] + 1, config['values'][2]))
        if config["rows_per_group"] == "random":
            arr = np.random.choice(arr, size=self.rows)
        else:
            pass
        
        if config['ordered']:
            arr = np.sort(arr)
        if config['percent_null'] > 0:
            num_missing = int(0.1 * self.rows)
            missing_indices = random.sample(range(self.rows), num_missing)
            arr = arr.astype(object)
            for idx in missing_indices:
                arr[idx] = None
 
        return arr
    
    def last_name(self, config: dict, data: pd.DataFrame):
        arr = np.random.choice(self.last_names['nachname'], size=self.rows)
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr
    
    def first_name(self, config: dict, data: pd.DataFrame):
        male_names = list(self.first_names[self.first_names['geschlecht'] == 'M']['vorname'])
        female_names = list(self.first_names[self.first_names['geschlecht'] != 'M']['vorname'])
        if "percent_male" in config:
            num_males = int(config["percent_male"] / 100 * self.rows)
            num_females = int(self.rows - num_males)
            arr = np.concatenate(
                (np.random.choice(male_names, size=num_males), 
                 np.random.choice(female_names, size=num_females))
            )
        elif "gender_field" in config:
            gender_column = config["gender_field"]
            male_identifier = config["gender_field_male_identfier"]
            
            # Generate names based on gender using list comprehension
            result = [
                np.random.choice(male_names) if gender == male_identifier else np.random.choice(female_names)
                for gender in data[gender_column]
            ]
            return result
        else:
            pass

        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr
    
    def canton(self, config: dict, data: pd.DataFrame):
        cantons = list(self.plz [config["field"]])
        arr = np.random.choice(cantons, size=self.rows)
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def plz4(self, config: dict, data: pd.DataFrame):
        if "canton_field" in config:
            plz_lookup_df = self.plz[['plz_postleitzahl', 'kanton_name', 'kanton_name_kurz']].drop_duplicates()
            cantons = data[config['canton_field']].drop_duplicates()
            df = data.copy()
            df['xxx'] = -99
            for canton in cantons:
                plz_for_canton = plz_lookup_df[plz_lookup_df[config['canton_field_version']]==canton]['plz_postleitzahl']
                if len(plz_for_canton) > 0:  # Check if there are PLZ codes for this canton
                    canton_mask = df[config['canton_field']] == canton
                    df.loc[canton_mask, 'xxx'] = np.random.choice(plz_for_canton, size=canton_mask.sum())
            arr = list(df['xxx'])
        else:
            arr = np.random.choice(list(self.plz['plz_postleitzahl']), size=self.rows)
        
        if 'percent_null' in config:
            arr = self.remove_values(np.array(arr), config['percent_null'])
        return arr

    
    def location(self, config: dict, data: pd.DataFrame):
        if 'plz' in config:
            df = data[[config['plz']]].copy()
            df['location'] = None
            # for eachrow, find location from plz: self.plz[self.plz[plz_postleitzahl == data[config['plz']].iloc[0]
            for idx, row in df.iterrows():
                plz_value = data[config['plz']].iloc[idx]
                matching_location = self.plz[self.plz['plz_postleitzahl'] == plz_value]['gemeinde_name'].iloc[0] if not pd.isna(plz_value) else None
                df.at[idx, 'location'] = matching_location
            arr = df['location'].to_numpy()
        else:
            if config['canton']=='list':
                df = self.plz[self.plz['gemeinde_name'].str.isin(config['canton'])]
            elif is_valid_canton(self.cantons, config['canton']):
                df = self.plz[self.plz['kanton_name_kurz']]
            else:
                df = self.plz
            arr = np.random.choice(df['gemeinde_name'], size=self.rows)
        if 'percent_null' in config:
            arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def address(self, config: dict):
        if config['canton']=='list':
            df = self.plz[self.plz['gemeinde_name'].isin(config['canton'])]
        elif is_valid_canton(self.cantons, config['canton']):
            df = self.plz[self.plz['kanton_name_kurz']]
        else:
            df = self.plz
        arr = np.random.choice(df['gemeinde_name'], size=self.rows)
        arr = self.remove_values(arr, config['percent_null'] )
        return arr
    
    def list_string(self, config: dict, data:pd.DataFrame):
        if "weights" in config:
            arr = np.random.choice(config["list"], self.rows, p=np.array(config["weights"])/np.sum(config["weights"]))
        else:
            arr = np.random.choice(config["list"], self.rows)
        arr = self.remove_values(arr, config['percent_null'] )
        return arr

    def streetname (self, config: dict, data:pd.DataFrame):
        if config["source"]==OGDBS:
            if "location_field" in config:
                pass
            if "location_field" in config:
                pass
            else:
                arr = np.random.choice(self.streets, size=self.rows) 
        else:
            arr = [fake.street_name() for _ in range(self.rows)]
            arr = np.array(arr)

        arr = self.remove_values(arr, config['percent_null'] )
        return arr
   
    def address(self, config: dict, data:pd.DataFrame):
        mapping = {
            "streetname": "str_name",
            "housenumber": "hausnr",
            "housenumber_addition": "hausnr_zus",
            "plz": "plz",
            "location": "ort",
        }

        fields = [v for k, v in mapping.items() if k in config["fields"]]
        df = self.addresses[fields].sample(n=self.rows, random_state=42)
        if 'hausnr' in df.columns:
            df['hausnr'] = df['hausnr'].astype(int)  
        df.columns = config['labels']
        return df
            
    def generate(self):
        data = pd.DataFrame()
        for key, config in self.columns.items():
            if config["function"] == "incremental_int":
                data[key] = self.incremental_int(config, data)
            elif config["function"] == "list_int":
                #data[key] = self.list_int(config, data)
                pass
            elif config["function"] == "random_int":
                data[key] = self.random_int(config, data)
            elif config["function"] == "random_float":
                data[key] = self.random_float(config, data)
            elif config["function"] == "normal_float":
                data[key] = self.normal_float(config, data)
            elif config["function"] == "list_int":
                data[key] = self.integer_grouped(config, data)
            elif config["function"] == "last_name":
                data[key] = self.last_name(config, data)
            elif config["function"] == "first_name":
                data[key] = self.first_name(config, data)
            elif config["function"] == "canton":
                data[key] = self.canton(config, data)
            elif config["function"] == "location":
                data[key] = self.location(config, data)
            elif config["function"] == "plz4":
                data[key] = self.plz4(config, data)
            elif config["function"] == "random_date":
                pass
            elif config["function"] == "list_string":
                data[key] = self.list_string(config, data)
            elif config["function"] == "streetname":
                data[key] = self.streetname(config, data)
            elif config["function"] == "address":
                df = self.address(config, data)                
                data = pd.concat([data.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            elif config["function"] == "ahv_nr":
                pass # df = self.ahv_nr(config, data)

        df = pd.DataFrame(data)
        with st.expander("Output"):
            st.write(df)
            data.to_csv(self.filename, index=False)
            

class DataMasker:
    def __init__(self, file_path, config_path):
        self.file_path = file_path
        with open(config_path, "r",  encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.data_in_df = pd.read_excel(file_path)
        self.data_out_df = self.data_in_df.copy()
        self.addresses = get_ogd_data(url=url_addresses, file=address_file)
        self.first_names = get_ogd_data(url=url_first_names, file=first_name_file)
        self.first_names['geschlecht'] = self.first_names['geschlecht'].replace({'W': 'F'})     
        self.last_names = get_ogd_data(url=url_last_names, file=last_name_file)

    def generate_ahv_number(self):
        """
        Generates a random, valid Swiss AHV number.

        Returns:
            str: A 13-digit Swiss AHV number in the format '756.XXXX.XXXX.XX'.
        """
        # Country code for Switzerland
        country_code = "756"
        unique_identifier = f"{random.randint(0, 99999999):08d}"

        # Combine the first 11 digits
        base_number = f"{country_code}{unique_identifier}"

        # Calculate the checksum using Modulo 11
        def calculate_checksum(number):
            weights = [1, 3]  # Alternating weights
            total = 0
            for i, digit in enumerate(number):
                weight = weights[i % 2]
                total += int(digit) * weight
            remainder = total % 11
            checksum = (11 - remainder) if remainder != 0 else 0
            return checksum if checksum < 10 else 0  # Replace 10 with 0

        # Compute the checksum
        checksum = calculate_checksum(base_number)

        # Format the AHV number
        ahv_number = f"{country_code}.{unique_identifier[:4]}.{unique_identifier[4:]}.{checksum:02d}"
        return ahv_number

    def save_pseudonymized(self, file_path):
        """
        Save the pseudonymized DataFrame to a file.

        :param file_path: File path to save the pseudonymized DataFrame
        """
        self.df.to_excel(file_path, index=False)

    def delete_rows_with_missing_values(self):
        """
        Delete rows with missing values in the DataFrame.
        """
        # not sure what I intended here 
        #for column, entry in self.config['columns'].items():
        #    if entry['function'] == None:
        #        self.data_in_df = self.data_in_df[self.data_in_df[column].notnull()]
        self.data_in_df = self.data_in_df.replace(r'^\s*$', np.nan, regex=True)
        # Drop rows that are completely empty
        self.data_in_df = self.data_in_df.dropna(how="all")
        
        self.data_out_df = self.data_in_df.copy()

    def pseudonymize(self):
        """
        Pseudonymize the columns in the DataFrame based on the configuration.

        :return: DataFrame with pseudonymized columns
        """

        def to_excel(df):
            # Use BytesIO to create a binary stream
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")
            output.seek(0)
            return output

        def format_columns():
            for column in self.data_in_df.columns:
                if self.config['columns'][column]['data_type'][:3].lower() == 'int':
                    s = pd.to_numeric(self.data_in_df[column], errors='coerce')      # blanks/strings -> NaN
                    s = s.replace([np.inf, -np.inf], pd.NA).astype('Int64')          # allow missing values
                    self.data_in_df[column] = s.astype(str) 

        def pseudonymize_all_columns():
            """Process all configured columns and return results."""
            results = {}
            for column in self.data_in_df.columns:
                if column in self.config['columns'] and self.config['columns'][column]['function']:
                    results[column] = self.pseudonymize_column(column)
            return results
        
        def display_processing_results(results):
            """Display column processing status in UI."""
            with st.expander("Spalten", expanded=True):
                result_icons = {True: "✔️", False: "❌"}
                for column, success in results.items():
                    st.write(f"Pseudonymisierung von Spalte: {column} {result_icons[success]}")
        
        def display_input_data():
            """Show the pseudonymized DataFrame."""
            with st.expander("Input"):
                st.dataframe(self.data_in_df)

        def display_output_data():
            """Show the pseudonymized DataFrame."""
            with st.expander("Output"):
                st.dataframe(self.data_out_df)
        
        def show_download_section():
            """Display success message and download button."""
            st.success(
                f"{len(self.data_out_df)} Zeilen wurden erfolgreich pseudonymisiert und die Datei kann heruntergeladen werden. Die Quelldatei wurde gelöscht."
            )
            st.download_button(
                label="Pseudonymisierte Datei herunterladen",
                data=to_excel(self.data_out_df),
                file_name="pseudonymized_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        format_columns()
        display_input_data()
        results = pseudonymize_all_columns()
        display_processing_results(results)
        display_output_data()
        show_download_section()
        

    def pseudonymize_column(self, column):
        """
        Pseudonymize a column in a DataFrame using a faker function.

        :param column: Column name to pseudonymize
        :return: DataFrame with pseudonymized column
        """
        ok = True
        config = self.config['columns'][column]
        if config['data_type'].lower()=='int':
            self.data_in_df[column] = self.data_in_df[column].astype(int) 
        faker_function = config["function"]
        if faker_function == "first_name":
            self.fake_first_names(column, config)
        elif faker_function == "last_name":
            self.fake_last_names(column, config)
        elif faker_function == "random_int":
            self.random_int(column, config)
        elif faker_function == "date_add_random_days":
            self.date_add_random_days(column, config)
        elif faker_function == "ahv_nr":
            self.ahv_nr(column, config)
        elif faker_function == "streetname_housenumber":
            self.random_address(column, config)
        elif faker_function == "phone_nr":
            self.phone_nr(column, config)
        elif faker_function == "formatted_numbers":
            self.formatted_number(column, config)
        elif faker_function == "email":
            self.email(column, config)
        elif faker_function == "street":
            self.street(column, config)
        elif faker_function == "house_number":
            self.house_number(column, config)
        elif faker_function == "hash":
            self.hash_value(column, config)
        elif faker_function == "delete":
            self.delete_column(column)
        elif faker_function == "shuffle_codes":
            self.shuffle_codes(column, config)
        elif faker_function == 'fill_with_random_distribution':
            self.fill_with_random_distribution(column, config)
        else:
            st.warning(f"'{faker_function}' ist keine bekannte faker-Funktion.")
            ok = False
        return ok

    @staticmethod
    def normalize_name(name):
        """
        Normalize a name by replacing special characters with their ASCII equivalents
        and removing any other unwanted characters.

        :param name: The input name (str).
        :return: The normalized name (str).
        """
        # Replace accented characters with their ASCII equivalents
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        normalized = re.sub(r"[ -]", ".", normalized)
        normalized = re.sub(r"[^a-zA-Z0-9.]", "", normalized)

        return normalized.lower()

    def hash_value(self, column, settings):
        """
        Create a hash value from concatenated columns specified in settings.

        :param column: The target column where the hash value will be stored
        :param settings: Dictionary containing:
                        - 'hashed_columns': List of column names to be concatenated and hashed
        """

        def calculate_hash(row):
            # Concatenate all specified columns, converting each to string
            concatenated_values = "".join(row[settings["hashed_columns"]].astype(str))
            # Generate SHA-256 hash of concatenated values
            return hashlib.sha256(concatenated_values.encode()).hexdigest()

        # Apply hash calculation to each row and store in output DataFrame
        self.data_out_df[column] = self.data_out_df.apply(calculate_hash, axis=1)

    def delete_column(self, column):
        """
        Delete a column from the output DataFrame.

        :param column: Name of the column to be deleted
        """
        if column in self.data_out_df.columns:
            self.data_out_df.drop(columns=[column], inplace=True)
        else:
            st.warning(f"Column '{column}' not found in DataFrame")

    def formatted_number(self, column, settings):
        """
        this methed is useful for phone and other formatted number,
        where you want to keep the pseudonym close to the original by
        only chaing the last n numbers ("replace_last_digits)

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "replace_last_digits": the number of digits at the end of the string to be replaced.
        """

        def replace_digits(number: str, digits_to_replace: int) -> str:
            result = list(str(number))
            digits_replaced = 0

            # Loop through the string in reverse order to find the last 4 digits
            for i in range(len(result) - 1, -1, -1):
                if result[i].isdigit():  # If the character is a digit
                    # Replace the digit with a random number
                    result[i] = str(random.randint(0, 9))
                    digits_replaced += 1

                # Stop once 4 digits are replaced
                if digits_replaced == digits_to_replace:
                    break

            # Join the list back into a string and return it
            return "".join(result)

        unique_numbers_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        unique_numbers_dict = {}

        for _, row in unique_numbers_df.iterrows():
            original_formatted_number = row[column]
            unique_numbers_dict[original_formatted_number] = replace_digits(
                original_formatted_number, settings["replace_last_digits"]
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: unique_numbers_dict[x] if pd.notnull(x) else x
        )

    def phone_nr(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """
        def scramble_phone_number(phone_number: str):
            """
            Scrambles the last 7 digits of a phone number while keeping the original format.
            Example:
                +41 79 174 2145 -> +41 79 528 9371
            """
            # Extract all digits
            if pd.isna(phone_number):
                return None
            else:    
                digits = re.findall(r'\d', phone_number)
                if len(digits) < 7:
                    raise ValueError("Phone number must have at least 7 digits.")
                
                # Split into prefix and part to scramble
                prefix_digits = digits[:-7]
                to_scramble = digits[-7:]
                
                # Scramble last 7 digits
                scrambled = random.sample(to_scramble, len(to_scramble))
                
                # Reassemble all digits
                new_digits = prefix_digits + scrambled
                
                # Replace digits in the original string one by one
                result = ""
                digit_index = 0
                for char in phone_number:
                    if char.isdigit():
                        result += new_digits[digit_index]
                        digit_index += 1
                    else:
                        result += char
                return str(result)


        self.data_out_df[column] = self.data_out_df[column].astype(str)
        unique_mobile_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        unique_mobile_df[column] = unique_mobile_df[column].apply(lambda x: str(x) if pd.notna(x) else x)
        mobile_dict = {}

        for _, row in unique_mobile_df.iterrows():
            original_mobile = row[column]
            scrambled_number = scramble_phone_number(original_mobile)
            mobile_dict[str(original_mobile)] = scrambled_number
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: mobile_dict.get(str(x), x) if pd.notnull(x) else x
        )

    def fill_with_random_distribution(self, column: str, settings: dict) -> bool:
        """
        Fills column `column` in df_out with random values drawn from a normal distribution
        based on the mean and standard deviation of the same column in df_in.

        Empty cells in df_in will remain empty in self.data_out_df

        Parameters:
        - df_in (pd.DataFrame): Input DataFrame containing source values.
        - df_out (pd.DataFrame): Output DataFrame where values will be filled.
        - column (str): Column name to be processed.

        Returns:
        - pd.DataFrame: Modified df_out with filled values.
        """
        try:
            # Ensure the column exists in both DataFrames
            if column not in self.data_in_df.columns or column not in self.data_out_df.columns:
                raise ValueError(f"Column '{column}' must exist in both DataFrames.")

            # Convert column to numeric, forcing errors to NaN
            self.data_in_df[column] = pd.to_numeric(self.data_in_df[column], errors='coerce')
            mean_value = self.data_in_df[column].mean()
            std_value = self.data_in_df[column].std()
            # Generate random values
            random_values = np.random.normal(mean_value, std_value, size=len(self.data_out_df))
            # Apply values while keeping NaNs from df_in
            self.data_out_df[column] = np.where(self.data_in_df[column].isna(), np.nan, random_values)
            self.data_out_df[column] = np.where(self.data_in_df[column].isna(), np.nan, np.round(random_values, settings['decimals']))
            return True
        except Exception as ex:
            st.warning(ex)
            return False

    def shuffle_codes(self, column, settings):
        try:
            # Get the unique values (codes) from the input DataFrame column
            unique_codes = self.data_in_df[column].dropna().unique()
            
            # Shuffle the unique codes using random.shuffle
            shuffled_codes = unique_codes.tolist()  # Convert to list for in-place shuffling
            random.shuffle(shuffled_codes)
            
            # Create a mapping from input codes to shuffled codes
            code_mapping = dict(zip(unique_codes, shuffled_codes))
            
            # Map the codes in the input DataFrame column to the shuffled codes
            self.data_out_df[column] = self.data_in_df[column].map(code_mapping)
            
            # Return True if no errors occurred
            return True
        
        except Exception as e:
            # Print any error and return False
            print(f"Error: {e}")
            return False

    def email(self, column, settings):
        """
        Replace emails only if the field is not null/empty.
        If first/last name fields are provided, generate name-based emails.
        Otherwise generate random emails. Keep null/empty cells unchanged.
        """

        providers = settings.get("providers", ["example.com"])

        def _is_filled(val) -> bool:
            # True if not NA and not just whitespace (for strings)
            if pd.isna(val):
                return False
            return (val.strip() != "") if isinstance(val, str) else True

        def get_fake_email_from_name(row):
            provider = random.choice(providers)
            fname_key = settings["first_name_field"]
            lname_key = settings["last_name_field"]

            fname_val = row.get(fname_key)
            lname_val = row.get(lname_key)

            if not (_is_filled(fname_val) and _is_filled(lname_val)):
                # Fallback: random email if names are missing/empty
                return fake.email()

            first_name = DataMasker.normalize_name(fname_val)
            last_name  = DataMasker.normalize_name(lname_val)
            return f"{first_name}.{last_name}@{provider}"

        if "first_name_field" in settings and "last_name_field" in settings:
            # Row-wise, because we need first/last name per row
            self.data_out_df[column] = self.data_out_df.apply(
                lambda row: get_fake_email_from_name(row) if _is_filled(row[column]) else row[column],
                axis=1,
            )
        else:
            # Vectorized map with safe conditional — keep null/empty as-is
            self.data_out_df[column] = self.data_in_df[column].map(
                lambda x: fake.email() if _is_filled(x) else x
            )

    def get_fake_address(self, addresses: list, settings: dict):
        """
        Get a fake address based on the specified location.

        :param ort: The location to generate a fake address for.
        :return: A fake address based on the specified location.
        """
        # limit choice either by ort or plz, if not available, use fakder.fake address
        if settings['source'] == 'ogd-bs':
            address = addresses.sample(1).iloc[0]
            street_housenr = f"{address['str_name']} {int(address['hausnr'])}"
            if pd.notnull(address["hausnr_zus"]) and address["hausnr_zus"] != "":
                street_housenr += f"{address['hausnr_zus']}"
            return street_housenr

    def get_fake_street(self, location_dict):
        """
        Get a fake address based on the specified location.

        :param ort: The location to generate a fake address for.
        :return: A fake address based on the specified location.
        """
        if not (
            location_dict["location_value"]
            in self.addresses[location_dict["location_code_col"]].values
        ):
            return fake.street_name()
        else:
            df = self.addresses[
                self.addresses[location_dict["location_code_col"]]
                == location_dict["location_value"]
            ]
            streets = df.sample(1).iloc[0]
            return streets["str_name"]

    def street(self, column, settings):
        """
        replaces street names by random street names from the same postal code

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        # Determine the range of days to add
        unique_street_fields = [column, settings["location_data_col"]]
        unique_street_fields = (
            self.data_in_df[unique_street_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        street_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_street_fields.iterrows():
            original_street = row[column]
            street_dict[original_street] = self.get_fake_street(
                {
                    "location_code_col": settings["location_code_col"],
                    "location_value": row[settings["location_data_col"]],
                }
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: street_dict[x] if pd.notnull(x) else x
        )

    def house_number(self, column, settings):
        """
        Gets a different random house number from the same street
        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        def get_random_housenumber(has_suffix):
            house_number = random.randint(1, 100)
            if has_suffix:
                suffix = random.choice(["a", "b", "c", "d", "e", "f"])
                return f"{house_number}{suffix}"
            return house_number

        # Generate aliases for unique names based on gender
        for index in self.data_out_df[self.data_out_df[column].notna()].index:
            with_suffix = random.random() < settings["frequency_suffix"]
            self.data_out_df.loc[index, column] = get_random_housenumber(with_suffix)

    def split_street_housenumber(self, address):
        """
        Split the street and house number from the combined string.

        :param street_housenumber: The combined street and house number.
        :return: The street and house number as separate strings.
        """
        address = address.strip()

        # Use a regular expression to split street and house number
        match = re.match(r"^(.*)\s+(\d+)$", address)

        if match:
            street = match.group(1).strip()  # The part before the last number
            house_number = match.group(2).strip()  # The numeric house number
            return (street, house_number)
        else:
            return (address, None)

    def change_house_number(self, address, location_dict):
        """
        Change the house number in the address to a random number from the same street. if there are no other house numbers
        in the same street, pick a random address from the same location.

        :param address: The original address.
        :param location_dict: A dictionary with configuration, including:
                            - "location_code_col": The column containing the postal code.
                            - "location_value": The postal code of the address.
        :return: The address with a random house number from the same street.
        """
        # Split the street and house number
        street, house_number = self.split_street_housenumber(address)
        if house_number is None:
            return address
        else:
            df = self.addresses[
                (self.addresses["str_name"] == street)
                & (
                    self.addresses[location_dict["location_code_col"]]
                    == location_dict["location_value"]
                )
            ]
            if len(df) > 1:
                random_address_in_street = df.sample(1).iloc[0]
                house_number = (
                    random_address_in_street["hausnr"]
                    if random_address_in_street["hausnr_zus"] is None
                    else random_address_in_street["hausnr"]
                    + random_address_in_street["hausnr_zus"]
                )
                return f"{street} {house_number}"
            else:
                address = self.get_fake_address(location_dict)
                return address

    def blur_address(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        # Determine the range of days to add
        unique_address_fields = settings["unique_address_fields"]
        unique_address_df = (
            self.data_in_df[unique_address_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        address_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_address_df.iterrows():
            original_address = row[column]
            address_dict[original_address] = self.change_house_number(
                original_address,
                {
                    "location_code_col": settings["location_code_col"],
                    "location_value": row[settings["location_data_col"]],
                },
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: address_dict[x] if pd.notnull(x) else x
        )

    def random_address(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        address_fields = [column]
        if "plz_field" in settings:
            address_fields.append(settings["plz_field"])
        elif "location_field" in settings:
            address_fields.append(settings["location_field"])
        unique_address_df = (
            self.data_in_df[address_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        address_dict = {}

        # Generate aliases for unique names based on gender
        for _, original_address in unique_address_df.iterrows():            
            if settings['source'] == 'ogd-bs':
                if "plz_field" in settings:
                    addresses = self.addresses[self.addresses['plz'] == original_address[settings["plz_field"]]]
                    if len(addresses) > 0:
                        random_adr = addresses.sample(1).iloc[0]
                        adr_str = f"{random_adr['str_name']} {int(random_adr['hausnr'])}{random_adr['hausnr_zus'] if random_adr['hausnr_zus'] else ''}"
                        address_dict[original_address[column]] = adr_str 
                    # if address not from basel
                    else: 
                        address_dict[original_address[column]] = fake.street_address()
                elif "location_field" in settings:
                    addresses = self.addresses[self.addresses['ort'] == original_address[settings["location_field"]]]
                    if len(addresses) > 0:
                        random_adr = addresses.sample(1).iloc[0]
                        adr_str = f"{random_adr['str_name']} {int(random_adr['hausnr'])}{random_adr['hausnr_zus'] if random_adr['hausnr_zus'] else ''}"
                        address_dict[original_address[column]] = adr_str 
                    # if address not from basel
                    else: 
                        address_dict[original_address[column]] = fake.street_address()
                else:
                    pass
            else:
                address_dict[original_address[column]] = fake.street_address()
            
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: address_dict[x] if pd.notnull(x) else x
        )

    def ahv_nr(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """
        # Determine the range of days to add

        unique_ahvnr_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        ahvnr_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_ahvnr_df.iterrows():
            original_ahvnr = row[column]
            ahvnr_dict[original_ahvnr] = self.generate_ahv_number()
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: ahvnr_dict[x] if pd.notnull(x) else x
        )

    def date_add_random_days(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """
        # Determine the range of days to add
        min_days = settings["lower"]
        max_days = settings["upper"]

        # Generate a list of random days to add
        random_days = [
            random.randint(min_days, max_days) for _ in range(len(self.data_in_df))
        ]

        unique_dates_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        date_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_dates_df.iterrows():
            original_date = row[column]
            if isinstance(original_date, str):
                original_date_dt = datetime.strptime(original_date, "%Y-%m-%dT%H:%M:%S")
                random_days = random.randint(min_days, max_days)
                # Add the random days to the base date
                new_date = original_date_dt + timedelta(days=random_days)
                date_dict[original_date] = new_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                date_dict[original_date] = original_date
        # Apply the generated aliases back to the DataFrame
        self.data_out_df[column] = self.data_in_df[column].map(date_dict)

    def random_int(self, column, faker_parameters):
        """
        Generate distinct random numbers for each row in the specified column of the DataFrame.

        :param column: The column to populate with random numbers.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_value": The minimum value for the random number.
                            - "max_value": The maximum value for the random number.
        """
        # Determine the range and ensure it's large enough for distinct numbers
        min_value = faker_parameters.get("min", 0)
        max_value = faker_parameters.get("max", 100)
        unique = faker_parameters.get("unique", False)
        # todo! unique random values
        unique_numbers_df = (
            self.data_in_df[[column]]
            .dropna(subset=[column])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        numbers_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_numbers_df.iterrows():
            original_name = row[column]
            if original_name:
                numbers_dict[original_name] = random.randint(min_value, max_value + 1)

        # Apply the generated aliases back to the DataFrame
        self.data_out_df[column] = self.data_in_df[column].map(numbers_dict)

    def detect_gender_from_firstname(self, first_name: str) -> str:
        name_row = self.first_names[self.first_names["vorname"] == first_name].iloc[0]
        if len(name_row) > 0:
            return name_row["geschlecht"]
        elif first_name.endswith("a"):
            return genderEnum.FEMALE.value
        else:
            return genderEnum.MALE.value

    def fake_first_names(self, column, settings: dict):
        """
        Generate fake first names based on gender for a specified column in the DataFrame.

        :param column: The column containing the original names.
        :param settings: A dictionary with configuration, including:
                        - "gender_field": The column indicating gender.
                        - "female": The value representing female in the gender column.
        """

        def get_fake_first_name(row, orig_first_name, settings: dict):
            if settings["source"] == "ogd-bs":
                if "gender_field" in settings:
                    # map data source gender code to internal m/w gender code
                    gender = (
                        "M"
                        if str(row[settings["gender_field"]]) == str(settings["gender_field_male_identfier"])
                        else "F"
                    )
                else:
                    gender = self.detect_gender_from_firstname(orig_first_name)

                return (
                    self.first_names[self.first_names["geschlecht"] == gender]
                    .sample(1)
                    .iloc[0]["vorname"]
                )
            else:
                if "gender_field" in settings:
                    return (
                        fake.first_name_female()
                        if row[settings["gender_field"]] == "F"
                        else fake.first_name_male()
                    )
                else:
                    return fake.first_name()

        if "gender_field" in settings:
            unique_names_df = (
                self.data_in_df[[column, settings["gender_field"]]]
                .drop_duplicates()
                .dropna(subset=[column])
                .reset_index(drop=True)
            )
        else:
            unique_names_df = (
                self.data_in_df[[column]]
                .drop_duplicates()
                .dropna(subset=[column])
                .reset_index(drop=True)
            )
        name_dict = {}

        for _, row in unique_names_df.iterrows():
            original_name = row[column]
            if original_name:
                name_dict[original_name] = get_fake_first_name(
                    row, original_name, settings
                )

        self.data_out_df[column] = self.data_in_df[column].map(name_dict)

    def fake_last_names(self, column, settings: dict):
        """
        Generate fake first names based on gender for a specified column in the DataFrame.

        :param column: The column containing the original names.
        :param settings: A dictionary with configuration, including:
                        - "gender_field": The column indicating gender.
                        - "female": The value representing female in the gender column.
        """

        def get_fake_last_name(row, settings):
            if "source" in settings and settings["source"] == "ogd-bs":
                return self.last_names.sample(1).iloc[0]["nachname"]
            else:
                return fake.last_name()
                

        # Extract unique name-gender combinations
        unique_names_df = (
            self.data_in_df[[column]]
            .dropna(subset=[column])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        name_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_names_df.iterrows():
            original_name = row[column]
            if original_name:
                name_dict[original_name] = get_fake_last_name(row, settings)

        # Apply the generated aliases back to the DataFrame
        self.data_out_df[column] = self.data_in_df[column].map(name_dict)

    def save_json_template(json_template, file_path):
        """
        Save the JSON template to a file.

        :param json_template: Dictionary template for JSON configuration
        :param file_path: File path to save the JSON template
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(json_template, file, ensure_ascii=False, indent=2)

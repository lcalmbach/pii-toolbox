# Pseudonymizer App

A user-friendly **Streamlit app** that enables you to pseudonymize sensitive data in **CSV** or **XLSX** files based on a customizable configuration file.

## Installation

## Features
- Upload **CSV** or **XLS/XLSX** files containing sensitive data.
- Provide a configuration file to define pseudonymization rules for each column.
- Pseudonymize the data interactively with options for review.
- Download the pseudonymized file for further use.

---

## How It Works

1. **Upload the Data File**:
   - Choose a **CSV** or **XLS/XLSX** file containing the data you want to pseudonymize.

2. **Upload the Configuration File**:
   - Provide a JSON configuration file that specifies:
     - Columns to pseudonymize.
     - The pseudonymization rules for each column (e.g., fake names, email addresses).

3. **Pseudonymization Process**:
   - The app applies the rules defined in the configuration file to the data.
   - Review and verify the pseudonymized output in the app.

4. **Download the Result**:
   - Download the pseudonymized file for secure use or further processing.

---

## Configuration File Format

The configuration file is a JSON file where each column in the dataset that requires pseudonymization is defined. Below is an example:

```json
{
    "student_id": {
        "pseudonymize": true,
        "not_null": true,
        "faker_function": "random_number",
        "faker_function_input_parameters": {"min_value": 200000, "max_value": 700000, "unique": true}
    },
    "student_name": {
        "pseudonymize": true,
        "faker_function": "last_name",
        "faker_function_input_parameters": {}
    },
    "student_gender": {
        "pseudonymize": false,
        "faker_function": null
    },
    "addresse": {
        "pseudonymize": true,
        "faker_function": "address",
        "faker_function_input_parameters": {"unique_address_fields": ["adress", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"}
    },
    ...
}
```

## Pseudonymizer functions

The following functions have been implemented and can be used as the keyword faker_function in the config file

### Integers

#### integer_random
Fills each cell in this column by a random number between min and max.  

| Parameter | Description |
| -------- | ----------- | 
| min | minium allowed value |
| max | maximum allowed value |
| unique | generate unique values |
| pct_null | randomly set (perc_null values * number of rows) cells to null after filling all rows|

Example:
```json
{
    "min": 200000, 
    "max": 700000, 
    "unique": true, 
    "pct_null": 0.1
}
```

#### integer_increment
Fills each cell in this column by incrementing number between starting from the minimum value up to minimum value + rows - 1. 

| Parameter | Description |
| -------- | ----------- | 
| start | start value |

Example:
```json
{
    "start": 10000
}
```

#### integer_normal
Fills each cell in this column by an integer value calculated from a normal distribution. Minimum and Maximum values can be set to prevent unrealistic value such as a negative length. Please note that by setting a minimum and maximum value, the result will no longer be normally distributed but may become biased. 

| Parameter | Description |
| -------- | ----------- | 
| mean | mean value of normal distribution |
| std | standard deviation |
| min | minimum allowed value, e.g. 0 prevents a length parameter to become negative |
| max | maximum allowed value, e.g. 220 for a parmaeter body_height prevents the generation of giants|

Example:
```json
{
    "mean": 90, 
    "std": 5, 
    "min": 0, 
    "max": 1000
}
```
### Strings
#### random_address
Replaces each occurrence of an address (street street number) by a random address of the same location. 

| Parameter | Description |
| -------- | ----------- | 
| unique_address_fields | list of fields defining the address |
| "location_code_col" | standard deviation |
| location_data_col | minmum allowed value |
| max | maximum allowed value |

```json
{
    "unique_address_fields": ["addresse", "postal_code"], 
    "location_code_col": "plz", 
    "location_data_col": "postal_code"
}
```

#### blur_address
Replaces each occurrence of an address (street street number) by a random address from the same street by switching the house number. This garantees that persons or objects stay geographically close. This function can only be used in the pseudonymizing mode, where existing values are replaced. for generating synthtic data from scratch, use the random_address function. 

| Parameter | Description |
| -------- | ----------- | 
| unique_address_fields | list of fields defining the address |
| "location_code_col" | standard deviation |
| location_data_col | minmum allowed value |
| max | maximum allowed value |

Example:
```json
{
    "min": 200000, 
    "max": 700000, 
    "unique": true, 
    "pct_null": 0.1
}
```

#### gender
fill all cells with cells with a gender code

| Parameter | Description |
| -------- | ----------- | 
| m | code for gender male |
| f | code for gender female |
| d | code for gender diverse, if null only male and female gender codes will be generated|
| perc_null | percent of rows fill with null value, default = 0 |

Example:
```json
{
    "m": "Herr", 
    "max": "Frau", 
    "pct_null": 0.1
}
```

#### code
Fill all cells with cells with a general codes

| Parameter | Description |
| -------- | ----------- | 
| codes | a list of codes |
| probabilities | a probability value for each code |
| perc_null | percent of rows fill with null value, default = 0 |

Example:
```json
{
    "codes": ["Basel", "Zürich", "Bern", "Genf"],
    "probabilities": [0.2, 0.4, 0.2, 0.2],  
    "pct_null": 0.1
}
```

#### first_name 
Replaces each occurrence of a first name by a random first name using the library faker or alternatively data from the first name OGD dataset [Vornamen der baselstädtischen Bevölkerung](https://data.bs.ch/explore/dataset/100129). if a column holds the gender value, you can have male and female names generated based on the gender value. 

| Parameter | Description |
| -------- | ----------- | 
| source | faker or bs: faker generates german firstnames, bs uses firstnames from the firstnames of the canton of Basel-STadt and therefore generates more realistic sounding results |
| gender_col | if a gender column exists, female and male names may be generated based on this column |
| perc_null | percent of rows fill with null value, default = 0 |

Example:
```json
{
    "m": "Herr", 
    "max": "Frau", 
    "pct_null": 0.1
}
```

#### last_name
Replaces each occurrence of a last name by a random last name using the library faker. 

| Parameter | Description |
| -------- | ----------- | 
| source | faker or bs: faker generates german firstnames, bs uses lastnames from the lastnames of the canton of Basel-Stadt and therefore generates more realistic sounding results |
| perc_null | percent of rows fill with null value, default = 0 |

Example:
```json
{
    "source": "bs", 
    "pct_null": 0.1
}
```

#### random_text
fills each cell with a random text. 

| Parameter | Description |
| -------- | ----------- | 
| min_sentences | minimum number of sentences default = 1 |
| max_sentences | maximum number of sentences, default = 4 |

Example:
```json
{
    "min_sentences": 1, 
    "max_sentences": 5
}
```
### Float numbers
#### float_random
fills each cell with a random float number. 

| Parameter | Description |
| -------- | ----------- | 
| min | minimum number |
| max | maximum number |

Example:
```json
{
    "min": 1.0, 
    "max": 5.5
}
```

#### float_normal
fills each cell with a random float number. 

| Parameter | Description |
| -------- | ----------- | 
| mean | mean of normal distribution |
| std | standard deviation of normal distribution |
| min | minimum value |
| max | maximum value |
| pct_null | percent of null values, default = 0 |

Example:
```json
{
    "mean": 8.3, 
    "std": 2.5,
    "min": 0,
    "max": 20
}
```

### Dates
#### date_random
#### date_normal
#### date_increment

### Boolean values
#### boolean
fills all cells with random values

| Parameter | Description |
| -------- | ----------- | 
| mean | mean of normal distribution |
| std | standard deviation of normal distribution |
| min | minimum value |
| max | maximum value |
| pct_null | percent of null values, default = 0 |

Example:
```json
{
    "true": "ja", 
    "false": "nein",
    "ratio_true": 0.8,
    "perc_null": 2
}
```



## Opendata
The addresses and street names are generated using a dataset from [data.bs](https://data.bs.ch/explore/dataset/100259) in order to generate familiar addresses. generating addresses requires a location field, which is either the location name or a postal code. If no postal code from Basel-Stadt Switzerland is provided, the faker.street() function is used to generate the street names.

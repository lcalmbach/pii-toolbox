# PII Toolbox

A user-friendly **Streamlit app** that enables you to pseudonymize sensitive data in **CSV** or **XLSX** files based on a customizable configuration file.

## Installation

## Features
- Upload **CSV** or **XLS/XLSX** files containing sensitive data.
- Generate a generic configuration file template based on the uploaded data.
- Provide a configuration file to define pseudonymization rules for each column.
- Pseudonymize the data interactively with options for review.
- Download the pseudonymized file for further use.
- Anonymize text data using the opensource [MS presidio](https://microsoft.github.io/presidio/) framework.

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
}
```

## Pseudonymizer functions

The pseudonymizer supports various functions to generate fake data. Here are some examples. a documentation of all functions is included in the GUI and the menu item Funktionen.


## Opendata
The addresses and street names are generated using a dataset from [data.bs](https://data.bs.ch/explore/dataset/100259) in order to generate familiar addresses. generating addresses requires a location field, which is either the location name or a postal code. If no postal code from Basel-Stadt Switzerland is provided, the faker.street() function is used to generate the street names.

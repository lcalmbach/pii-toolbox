# PII Toolbox

A user-friendly **Streamlit app** that enables you to anonymize sensitive data in **CSV** or **XLSX** files based on a customizable configuration file.

## Installation

## Features
- Upload **CSV** or **XLS/XLSX** files containing sensitive data.
- Generate a generic configuration file template based on the uploaded data.
- Provide a configuration file to define anonymization rules for each column.
- Anonymize the data interactively with options for review.
- Download the anonymized file for further use.
- Anonymize text data using the opensource [MS presidio](https://microsoft.github.io/presidio/) framework.

---

## How It Works

1. **Upload the Data File**:
   - Choose a **CSV** or **XLS/XLSX** file containing the data you want to anonymize.

2. **Upload the Configuration File**:
   - Provide a JSON configuration file that specifies:
     - Columns to anonymize.
     - The anonymization rules for each column (e.g., fake names, email addresses).

3. **Anonymizsation Process**:
   - The app applies the rules defined in the configuration file to the data.
   - Review and verify the anonymized output in the app.

4. **Download the Result**:
   - Download the anonymized file for secure use or further processing.

---

## Configuration File Format

The configuration file is a JSON file where each column in the dataset that requires anonymization is defined. Below is an example:

```json
{
    "student_id": {
        "anonymize": true,
        "not_null": true,
        "faker_function": "random_number",
        "faker_function_input_parameters": {"min_value": 200000, "max_value": 700000, "unique": true}
    },
    "student_name": {
        "anonymize": true,
        "faker_function": "last_name",
        "faker_function_input_parameters": {}
    },
    "student_gender": {
        "anonymize": false,
        "faker_function": null
    },
    "addresse": {
        "anonymize": true,
        "faker_function": "address",
        "faker_function_input_parameters": {"unique_address_fields": ["adress", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"}
    },
}
```

A complete list of available functions can be found under the menu option `Functionen`
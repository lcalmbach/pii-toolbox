from streamlit_option_menu import option_menu
import streamlit as st
import anonymizer_data
import anonymizer_text
import texts
import json
from pathlib import Path
import mimetypes
from enum import Enum
import pandas as pd
import chardet
import fitz

__version__ = "0.0.6"
__author__ = "lukas.calmbach@bs.ch"
AUTHOR_NAME = 'Lukas Calmbach'
VERSION_DATE = "2025-02-05"
APP_NAME = "PII Toolbox"
app_emoji = "🗣️"

data_folder = "./src/data/"
file_path = None # file_path__base + extension
json_file_path = "./src/data/config.json"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=app_emoji,
    layout="wide",
)

menu_options = [
    {"Über die App": "info"},
    {"Testdaten-Konfiguration erstellen": "gear"},
    {"Bestehende Datei Anonymisieren": "file-earmark-excel"},
    {"Testdaten erstellen": "file-earmark-spreadsheet"},
    {"Dokument Anonymisieren": "file-earmark-font"},
    {"Anleitung": "book"},
]
class MenuIndex(Enum):
    ABOUT = 0
    CREATE_CONFIG = 1
    ANONYMIZE_TABLE = 2
    CREATE_TEST_DATA_TABLE = 3
    ANONYMIZE_TEXT = 4
    HELP = 5

def get_file_path(filename_only: str)->str:
    return str(Path(data_folder, filename_only))

def display_app_info():
    """
    Zeigt die Infos zur Applikations in einer Info-box in der Sidebar an.
    """
    text = f"""
    <style>
        #appinfo {{
        font-size: 11px;
        background-color: lightblue;
        padding-top: 10px;
        padding-right: 10px;
        padding-bottom: 10px;
        padding-left: 10px;
        border-radius: 10px;
    }}
    </style>
    <div id="appinfo">
    {APP_NAME}<br>
    Version: {__version__} ({VERSION_DATE})<br>
    Entwicklung: Statistisches Amt Basel-Stadt<br>
    Kontakt: <a href="{__author__}">{AUTHOR_NAME}</a>
    </div>
    """
    st.sidebar.markdown(text, unsafe_allow_html=True)

def generate_config_from_file(file_path):
    with st.spinner("Prozess läuft..."):
        st.session_state.config_file = anonymizer_data.generate_json_config(file_path)
        mime_type, _ = mimetypes.guess_type(st.session_state.config_file)
        if st.session_state.config_file:
            st.success('Die Konfiguration wurde erfolgreich generiert')
            st.download_button(
                label="Konfiguration herunterladen",
                data=open(st.session_state.config_file, "rb").read(),
                file_name =  Path(st.session_state.config_file).name,
                mime = mime_type
            )                  
        else:
            st.warning('Bei der Generierung des Konfigurationsdatei ist ein Fehler aufgetreten.')

def create_fields():
    if 'config_file' not in st.session_state:
        st.session_state.config_file = st.text_input("Konfig Datei")
    if 'config_file' in st.session_state:
        st.session_state.config = json.read
        with open(st.session_state.config_file, "r", encoding="utf-8") as f:
            st.session_state.config = json.load(f)
            # fields = 
        
def init():
    st.session_state.row_number = 100
    st.session_state.format = 'csv'
    st.session_state.encoding = 'utf-8'
    st.session_state.filename = 'output.csv'
    st.session_state.separator = ';'


# menu items
def about():
    with open("./src/docs/info.md", "r", encoding="utf8") as file:
        anleitung_content = file.read()
        st.markdown(anleitung_content, unsafe_allow_html=True)

def create_config():
    with st.expander('Hilfe'):
        st.markdown(texts.konfig_erstellen)
    options = ["Beispiel-Datei hochladen", "Felder manuell erfassen"]
    mode = st.radio("Eingabemethode wählen:", options)
    if options.index(mode) == 0:
        uploaded_file = st.file_uploader("Auswahl Daten-Datei")
        if uploaded_file:
            filename_only = Path(uploaded_file.name).stem
            file_path = get_file_path(filename_only + Path(uploaded_file.name).suffix)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            if st.button("Konfigurations-Datei erstellen"):
                generate_config_from_file(file_path)    
    else:
        st.info("Diese Option steht noch nicht zur Verfügung")

def anonymize_table():
    with st.expander("Hilfe"):
        st.markdown(texts.info_pseudonymizer)
    uploaded_file = st.file_uploader("Auswahl Daten-Datei")
    uploaded_config = st.file_uploader("Auswahl Konfigurations-Datei")
    if uploaded_config:
        filename_only = Path(uploaded_file.name).stem
        json_file_path = get_file_path(filename_only + '.json')
        with open(json_file_path, "wb") as f:
            f.write(uploaded_config.getvalue())
    if uploaded_file:
        filename_only = Path(uploaded_file.name).stem
        file_path = get_file_path(filename_only + Path(uploaded_file.name).suffix)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    if uploaded_file and uploaded_config:
        if st.button("👤 Anonymisierung starten"):
            with st.spinner("Anonymisierung läuft..."):
                p = anonymizer_data.DataMasker(file_path, json_file_path)
                p.pseudonymize()

def create_test_data_table():
    with st.expander('Hilfe'):
        st.markdown(texts.testdaten_erstellen)
        
    uploaded_config = st.file_uploader("Auswahl Konfigurations-Datei")
    if uploaded_config:
        with open(json_file_path, "wb") as f:
            f.write(uploaded_config.getvalue())
        json_data = json.load(uploaded_config)
        st.session_state.row_number = json_data['rows']
        st.session_state.format = json_data['format'].lower()
        st.session_state.encoding = json_data['encoding'].lower()
        st.session_state.filename = json_data['filename']
        st.session_state.separator = json_data['separator'].lower()
        with st.expander("Konfiguration"):
            st.json(json_data)
        
    cols = st.columns([2,2,2,1])
    st.session_state.filename = st.text_input("Dateiname:", value=st.session_state.filename)
    with cols[0]:
        st.session_state.row_number = st.number_input("Anzahl Datensätze:", min_value=1, value=st.session_state.row_number)
    with cols[1]:
        options=["csv", "excel", "json"]
        sel_index = options.index(st.session_state.format)
        st.session_state.format = st.selectbox("Format:", options, index=sel_index)
    with cols[2]:
        options = ["utf-8", "iso-8859-1", "ansi", "windows-1252", "ascii"]
        sel_index = options.index(st.session_state.encoding)
        st.session_state.encoding = st.selectbox("Encoding:", options,index=sel_index)
    with cols[3]:
        options = [";", ",", "tab"]
        sel_index = options.index(st.session_state.separator)
        st.session_state.separator = st.selectbox("Separator:", options, index=sel_index)
    
    if st.button("👤 Testdaten Generieren starten"):
        with st.spinner("Generierung von Testdaten läuft..."):
            p = anonymizer_data.DataGenerator(json_data)
            p.generate()
            st.success('Die Testdaten wurden erfolgreich generiert und können heruntergeladen werden')

import io
from pathlib import Path
import chardet
import pandas as pd
import fitz  # PyMuPDF
import streamlit as st

def anonymize_text():
    def extract_text_from_file(_uploaded_file):
        file_extension = Path(_uploaded_file.name).suffix.lower()

        valid_extensions = {'.pdf', '.txt', '.tab', '.csv'}
        if file_extension not in valid_extensions:
            raise ValueError(f"Invalid file type. Supported: {', '.join(sorted(valid_extensions))}")

        try:
            if file_extension == '.pdf':
                # Read bytes for PyMuPDF
                _uploaded_file.seek(0)
                pdf_reader = fitz.open(stream=_uploaded_file.read(), filetype="pdf")
                parts = []
                for page in pdf_reader:
                    # get_text() == get_text("text")
                    parts.append(page.get_text())
                text = "\n".join(parts)

            elif file_extension == '.txt':
                _uploaded_file.seek(0)
                raw_data = _uploaded_file.read()
                enc = chardet.detect(raw_data).get('encoding') or 'utf-8'
                text = raw_data.decode(enc, errors='replace')

            elif file_extension in ('.csv', '.tab'):
                _uploaded_file.seek(0)
                sep = '\t' if file_extension == '.tab' else ','
                # If you want to auto-detect encoding for CSV/TAB too:
                raw = _uploaded_file.read()
                enc = chardet.detect(raw).get('encoding') or 'utf-8'
                _uploaded_file = io.StringIO(raw.decode(enc, errors='replace'))
                df = pd.read_csv(_uploaded_file, sep=sep)
                # Serialize table to readable text (not CSV again)
                text = df.to_string(index=False)

        except Exception as e:
            raise Exception(f"Error processing {file_extension} file: {e}")

        return text.strip()

    with st.expander('Hilfe'):
        st.markdown("…dein Hilfetext…")

    options = ["Interaktive Texteingabe", "Dokument hochladen"]
    mode = st.radio("Eingabemethode wählen:", options)

    if options.index(mode) == 0:
        text = st.text_area("Text")
        if st.button('Text Anonymisieren'):
            p = anonymizer_text.TextAnonymizer(text)
            st.write(p.anonymize())
    else:
        uploaded_file = st.file_uploader("Datei hochladen", type=["pdf", "txt", "csv", "tab"])
        if uploaded_file:
            extracted = extract_text_from_file(uploaded_file)
            st.session_state.text = extracted
            st.text_area("Extrahierter Text", extracted, height=300)
            if st.button('Text Anonymisieren'):
                p = anonymizer_text.TextAnonymizer(extracted)
                st.write(p.anonymize())


        
def main():
    if "filename" not in st.session_state:
        init()
    menu_labels = [list(item.keys())[0] for item in menu_options]
    menu_icons = [list(d.values())[0] for d in menu_options]
    with st.sidebar:
        st.title("👤 PII Toolbox")
        menu_action = option_menu(
            None,
            options=menu_labels,
            icons=menu_icons,
            menu_icon="cast",
            default_index=0,
        )
    st.subheader(menu_action)
    if menu_labels.index(menu_action) == MenuIndex.ABOUT.value:
        about()
    elif menu_labels.index(menu_action) == MenuIndex.CREATE_CONFIG.value:
       create_config()
    elif menu_labels.index(menu_action) == MenuIndex.ANONYMIZE_TABLE.value:
        anonymize_table()
    elif menu_labels.index(menu_action) == MenuIndex.CREATE_TEST_DATA_TABLE.value:
        create_test_data_table()
    elif menu_labels.index(menu_action) == MenuIndex.ANONYMIZE_TEXT.value:
        anonymize_text()
    elif menu_labels.index(menu_action) == MenuIndex.HELP.value:
        # open anleitung file
        with open("./src/docs/anleitung.md", "r", encoding="utf8") as file:
            anleitung_content = file.read()
            st.markdown(anleitung_content, unsafe_allow_html=True)

    display_app_info()

if __name__ == "__main__":
    main()
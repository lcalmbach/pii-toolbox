import io
import streamlit as st
from streamlit_option_menu import option_menu
import json
from pathlib import Path
import mimetypes
from enum import Enum
import pandas as pd
import chardet
import fitz
import logging
import re

from functions import show_functions
import anonymizer_data
import anonymizer_text
import texts

__version__ = "0.0.8"
__author__ = "lukas.calmbach@bs.ch"
AUTHOR_NAME = 'Lukas Calmbach'
VERSION_DATE = "2025-10-31"
APP_NAME = "PII Toolbox"
app_emoji = "🗣️"

file_path = None # file_path__base + extension
input_folder = Path("./src/input/")
output_folder = Path("./src/output/")
demo_folder = Path("./src/demo")
json_file_path = input_folder / "config.json"
demo_anonymize_data = demo_folder / "anonymize.xlsx"
demo_json = demo_folder / "anonymize.json"
demo_pdf = demo_folder / "demo_pdf.pdf"

# setup basic logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title=APP_NAME,
    page_icon=app_emoji,
    layout="wide",
)

menu_options = [
    {"Über die App": "info"},
    {"Testdaten-Konfiguration generieren": "gear"},
    {"Bestehende Datei Anonymisieren": "file-earmark-excel"},
    {"Testdaten generieren": "file-earmark-spreadsheet"},
    {"Dokument Anonymisieren": "file-earmark-font"},
    {"Funktionen": "book"},
]
class MenuIndex(Enum):
    ABOUT = 0
    CREATE_CONFIG = 1
    ANONYMIZE_TABLE = 2
    CREATE_TEST_DATA_TABLE = 3
    ANONYMIZE_TEXT = 4
    FUNCTIONS = 5

def get_file_path(filename_only: str)->str:
    return str(Path(input_folder, filename_only))

# Helpers: sanitize filename and save uploaded files into input_folder
def safe_filename(name: str) -> str:
    # keep base name only
    name = Path(name).name
    # replace unwanted chars
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    return safe

def save_uploaded_to_input(uploaded, target_folder: Path) -> Path:
    """Save UploadedFile or BytesIO to input_folder securely and return Path."""
    try:
        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)
        raw_name = getattr(uploaded, 'name', 'uploaded_file')
        filename = safe_filename(raw_name)
        target_path = target_folder / filename
        # get bytes
        try:
            content = uploaded.getvalue()
        except Exception:
            uploaded.seek(0)
            content = uploaded.read()
        with open(target_path, 'wb') as f:
            f.write(content)
        logging.info(f"Saved uploaded file to {target_path}")
        return target_path
    except Exception as e:
        logging.exception('Fehler beim Speichern der hochgeladenen Datei')
        st.error(f"Fehler beim Speichern der Datei: {e}")
        return None

def load_demo_bytes(path: Path) -> io.BytesIO:
    if path.exists():
        try:
            with open(path, 'rb') as f:
                b = f.read()
            bio = io.BytesIO(b)
            bio.name = path.name
            return bio
        except Exception as e:
            logging.exception('Fehler beim Laden der Demo-Datei')
            st.error(f"Fehler beim Laden der Demo-Datei: {e}")
    return None

def preview_table_file(path: Path, max_rows: int = 10):
    """Try to load a table and show a small preview."""
    if not path or not Path(path).exists():
        return
    try:
        p = Path(path)
        if p.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(p)
        else:
            # try to detect encoding
            raw = open(p, 'rb').read()
            enc = chardet.detect(raw).get('encoding') or 'utf-8'
            df = pd.read_csv(io.StringIO(raw.decode(enc, errors='replace')))
        with st.expander(f"**Vorschau ({p.name}) - erste {max_rows} Zeilen:**"):
            st.dataframe(df.head(max_rows))
    except Exception as e:
        logging.exception('Preview fehlgeschlagen')
        st.warning(f"Vorschau konnte nicht erstellt werden: {e}")

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
    st.session_state.uploaded_file_path = None
    st.session_state.uploaded_config_path = None
    st.session_state.config_json = None
    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)    


def create_config():
    with st.expander('Hilfe'):
        st.markdown(texts.konfig_erstellen)
    options = ["Beispiel-Datei hochladen", "Felder manuell erfassen"]
    mode = st.radio("Eingabemethode wählen:", options)
    if options.index(mode) == 0:
        uploaded_file = st.file_uploader("Auswahl Daten-Datei")
        if uploaded_file:
            saved = save_uploaded_to_input(uploaded_file, input_folder)
            if saved:
                st.success(f"Datei gespeichert: {saved.name}")
                preview_table_file(saved)
                if st.button("Konfigurations-Datei erstellen"):
                    generate_config_from_file(str(saved))
    else:
        st.info("Diese Option steht noch nicht zur Verfügung")


def anonymize_table():
    with st.expander("Hilfe"):
        st.markdown(texts.info_anonymizer)

    # Demo toggle fallback
    if hasattr(st, 'toggle'):
        demo = st.toggle("Demo-Modus", value=False, help="Bei aktiviertem Demo-Modus werden Beispiel-Dateien verwendet und Upload-Felder nicht angezeigt.")
    else:
        demo = st.checkbox("Demo-Modus", value=False, help="Bei aktiviertem Demo-Modus werden Beispiel-Dateien verwendet und Upload-Felder nicht angezeigt.")

    uploaded_file = None
    uploaded_config = None

    if demo:
        demo_file = load_demo_bytes(demo_anonymize_data)
        demo_cfg = load_demo_bytes(demo_json)
        if demo_file and demo_cfg:
            uploaded_file = demo_file
            uploaded_config = demo_cfg
            st.success("Demo-Dateien geladen")
        else:
            st.warning(f"Demo-Dateien nicht gefunden unter {demo_folder}.")
    else:
        uploaded_file = st.file_uploader("Auswahl Daten-Datei")
        uploaded_config = st.file_uploader("Auswahl Konfigurations-Datei")

    # Save uploaded files to input_folder and preview
    file_path = None
    json_file_path = None
    if uploaded_file:
        saved = save_uploaded_to_input(uploaded_file, input_folder)
        if saved:
            st.session_state.uploaded_file_path = str(saved)
            preview_table_file(saved)
            file_path = str(saved)
    if uploaded_config:
        saved_cfg = save_uploaded_to_input(uploaded_config, input_folder)
        if saved_cfg:
            st.session_state.uploaded_config_path = str(saved_cfg)
            # try to load JSON for immediate display
            try:
                with open(saved_cfg, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                with st.expander('Konfigurationsvorschau', expanded=True):
                    st.write(cfg)
                st.session_state.config_json = cfg
            except Exception as e:
                st.warning(f"Konfigurationsdatei konnte nicht gelesen werden: {e}")
            json_file_path = str(saved_cfg)

    # Only show start button if both files present
    if file_path and json_file_path:
        if st.button("👤 Anonymisierung starten"):
            with st.spinner("Anonymisierung läuft..."):
                try:
                    p = anonymizer_data.DataMasker(file_path, json_file_path)
                    p.anonymize()
                    st.success('Anonymisierung abgeschlossen')
                except Exception as e:
                    logging.exception('Fehler bei Anonymisierung')
                    st.error(f"An Fehler ist aufgetreten: {e}")
    else:
        st.info("Bitte Datendatei und Konfigurationsdatei bereitstellen, um die Anonymisierung zu starten.")


def create_test_data_table():
    with st.expander('Hilfe'):
        st.markdown(texts.testdaten_erstellen)

    if hasattr(st, "toggle"):
        demo = st.toggle("Demo-Modus", value=False, help="Bei aktiviertem Demo-Modus wird die Demo-Konfiguration create_table.json verwendet.")
    else:
        demo = st.checkbox("Demo-Modus", value=False, help="Bei aktiviertem Demo-Modus wird die Demo-Konfiguration create_table.json verwendet.")

    uploaded_config = None
    if demo:
        demo_json_path = demo_folder / "create_table.json"
        demo_cfg = load_demo_bytes(demo_json_path)
        if demo_cfg:
            uploaded_config = demo_cfg
            st.success("Demo-Konfiguration geladen")
        else:
            st.warning(f"Demo-Konfiguration nicht gefunden unter {demo_folder}.")
    else:
        uploaded_config = st.file_uploader("Auswahl Konfigurations-Datei")

    json_data = None
    if uploaded_config:
        saved_cfg = save_uploaded_to_input(uploaded_config, input_folder)
        if saved_cfg:
            try:
                with open(saved_cfg, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                st.error(f"Konfigurationsdatei konnte nicht gelesen werden: {e}")

    if json_data:
        with st.expander("Konfigurations-Datei"):
            st.write(json_data)
        # Übernehme Werte in den Session-State
        st.session_state.row_number = json_data.get('rows', st.session_state.get('row_number', 100))
        st.session_state.format = json_data.get('format', st.session_state.get('format', 'csv')).lower()
        st.session_state.encoding = json_data.get('encoding', st.session_state.get('encoding', 'utf-8')).lower()
        st.session_state.filename = json_data.get('filename', st.session_state.get('filename', 'output.csv'))
        st.session_state.separator = json_data.get('separator', st.session_state.get('separator', ';')).lower()

        cols = st.columns([2,10])
        with cols[0]:
            json_data['rows'] = st.number_input("Anzahl Datensätze:", min_value=1, value=st.session_state.row_number)
        with cols[1]:
            json_data['filename'] = st.text_input("Dateiname:", value=st.session_state.filename)
    
        if st.button("👤 Testdaten Generieren starten"):
            with st.spinner("Generierung von Testdaten läuft..."):
                try:
                    p = anonymizer_data.DataGenerator(json_data)
                    p.generate()
                    st.success('Die Testdaten wurden erfolgreich generiert und können heruntergeladen werden')
                except Exception as e:
                    logging.exception('Fehler bei Testdaten-Generierung')
                    st.error(f"Fehler bei der Generierung: {e}")


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
        st.markdown(texts.anonymize_texts)

    options = ["Interaktive Texteingabe", "Dokument hochladen", "Demo"]
    mode = st.radio("Eingabemethode wählen:", options)

    def display_and_anonymize(extracted_text: str, button_key: str):
        st.session_state.text = extracted_text
        st.text_area("Extrahierter Text", extracted_text, height=300)
        if st.button('Text Anonymisieren', key=button_key):
            p = anonymizer_text.TextAnonymizer(extracted_text)
            st.write(p.anonymize())

    if mode == options[0]:
        # Interactive input
        text = st.text_area("Text")
        if st.button('Text Anonymisieren', key='anon_interactive'):
            p = anonymizer_text.TextAnonymizer(text)
            st.write(p.anonymize())

    elif mode == options[1]:
        # Uploaded document
        uploaded_file = st.file_uploader("Datei hochladen", type=["pdf", "txt", "csv", "tab"])
        if uploaded_file:
            extracted = extract_text_from_file(uploaded_file)
            display_and_anonymize(extracted, button_key='anon_upload')

    else:
        # Demo: load demo PDF and reuse same display logic
        if demo_pdf.exists():
            with open(demo_pdf, "rb") as f:
                pdf_bytes = f.read()
            demo_file = io.BytesIO(pdf_bytes)
            demo_file.name = demo_pdf.name
            st.success("Demo-Datei geladen")
            extracted = extract_text_from_file(demo_file)
            display_and_anonymize(extracted, button_key='anon_demo')
        else:
            st.warning(f"Demo-Datei nicht gefunden unter {demo_folder}. Bitte lege demo_pdf.pdf dort ab.")

        
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
        st.markdown(texts.app_info)
    elif menu_labels.index(menu_action) == MenuIndex.CREATE_CONFIG.value:
       create_config()
    elif menu_labels.index(menu_action) == MenuIndex.ANONYMIZE_TABLE.value:
        anonymize_table()
    elif menu_labels.index(menu_action) == MenuIndex.CREATE_TEST_DATA_TABLE.value:
        create_test_data_table()
    elif menu_labels.index(menu_action) == MenuIndex.ANONYMIZE_TEXT.value:
        anonymize_text()
    elif menu_labels.index(menu_action) == MenuIndex.FUNCTIONS.value:
        show_functions()

    display_app_info()

if __name__ == "__main__":
    main()
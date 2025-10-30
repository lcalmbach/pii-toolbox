import streamlit as st
from all_functions import function_dict
import json
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

def show_functions():
    def show_function(key, value):
        st.subheader(key)
        st.write(value.get('description', ''))

        params = value.get('parameters', [])
        helps = value.get('help', [])
        defaults = value.get('defaults', [])

        if params:
            st.markdown("**Parameter:**")
            table = "| Parameter | Standardwert | Beschreibung |\n|---|---|---|\n"
            for i, par in enumerate(params):
                default = json.dumps(defaults[i], ensure_ascii=False) if i < len(defaults) else ""
                help_text = helps[i] if i < len(helps) else ""
                table += f"| {par} | {default} | {help_text} |\n"
            st.markdown(table)

        st.markdown("**Beispiel:**")
        example = value.get('example', '') or ''
        # Versuche, gültiges JSON schön zu formatieren; falls das fehlschlägt, rohen Text anzeigen
        formatted = example
        try:
            # Wenn das Beispiel ein JSON-Fragment ist, versuche es zu parsen; sonst mit geschweiften Klammern umschliessen
            trimmed = example.strip()
            if trimmed.startswith('{') or trimmed.startswith('['):
                obj = json.loads(trimmed)
                formatted = json.dumps(obj, indent=2, ensure_ascii=False)
            else:
                # Versuch, mit geschweiften Klammern gültiges JSON zu erzeugen
                obj = json.loads('{' + trimmed + '}')
                formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            formatted = example.strip()

        st.code(formatted, language='json')

    def show_list():
        for key, value in config.items():
            show_function(key, value)
            st.divider()

    def show_selection_form():
        # Erzeuge ein DataFrame mit Funktion und Beschreibung und zeige es als AgGrid
        names = list(config.keys())
        descriptions = [config[n].get('description', '') for n in names]
        df = pd.DataFrame({"Funktion": names, "Beschreibung": descriptions})

        st.markdown("**Verfügbare Funktionen (Übersicht):**")

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection(selection_mode='single', use_checkbox=False)
        gb.configure_default_column(sortable=True, filter=True)
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            height=300,
        )

        selected_rows = grid_response.get('selected_rows', [])
        # Unterstützung für verschiedene Rückgabeformate von AgGrid: Liste von dicts, DataFrame oder dict
        selected_name = None
        if selected_rows is None:
            selected_rows = []

        if hasattr(selected_rows, 'iloc'):
            # pandas DataFrame oder Series
            try:
                selected_name = selected_rows.iloc[0]['Funktion']
            except Exception:
                try:
                    selected_name = selected_rows.iloc[0].get('Funktion')
                except Exception:
                    selected_name = None
        elif isinstance(selected_rows, dict):
            selected_name = selected_rows.get('Funktion')

        if selected_name:
            st.session_state['selected_function'] = selected_name
        selected = st.session_state.get('selected_function')
        if selected:
            st.markdown('---')
            show_function(selected, config[selected])
        else:
            st.info("Wähle eine Zeile im Grid aus, um Details anzuzeigen.")

    config = function_dict
    options = ["Liste aller Funktionen", "Selektion einzelne Funktion"]
    mode = st.radio("Anzeige", options=options)
    if options.index(mode) == 0:
        show_list()
    else:
        show_selection_form()




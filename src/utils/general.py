import streamlit as st
import pandas as pd
import io
from pathlib import Path

def show_download_section(data, filename: str = None, file_type: str = None):
    """Display success message and suitable download button(s).

    Parameters
    - data: pandas.DataFrame, bytes/bytearray, str, or file-like object (.read())
    - filename: optional filename to use for download (recommended to include extension)
    - file_type: optional hint for file type ('xlsx','csv','txt')
    """
    # If a DataFrame is provided, offer a single download (CSV or Excel) based on filename/file_type
    if isinstance(data, pd.DataFrame):
        df = data

        # Decide desired output format: prefer explicit file_type, then filename extension, default to CSV
        desired = None
        if file_type:
            ft = file_type.lower()
            if ft in ('xlsx', 'xls', 'excel'):
                desired = 'xlsx'
            elif ft in ('csv',):
                desired = 'csv'
        if desired is None and filename:
            low = filename.lower()
            if low.endswith(('.xlsx', '.xls')):
                desired = 'xlsx'
            elif low.endswith('.csv'):
                desired = 'csv'
        if desired is None:
            desired = 'csv'

        if desired == 'xlsx':
            # Excel in-memory
            excel_buffer = io.BytesIO()
            try:
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                excel_buffer.seek(0)
                excel_name = filename or "anonymized_data.xlsx"
                if not excel_name.lower().endswith(('.xlsx', '.xls')):
                    excel_name = Path(excel_name).stem + '.xlsx'
                st.download_button(
                    label="Datei herunterladen (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                # Fallback to CSV if Excel writer not available
                st.warning("Excel-Download nicht verfügbar (abhängige Bibliothek fehlt). Es wird stattdessen CSV angeboten.")
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                csv_name = (Path(filename).stem if filename else "anonymized_data") + ".csv"
                st.download_button(
                    label="Datei herunterladen (CSV)",
                    data=csv_bytes,
                    file_name=csv_name,
                    mime="text/csv",
                )
        else:
            # CSV
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            csv_name = (Path(filename).stem if filename else "anonymized_data") + ".csv"
            st.download_button(
                label="Datei herunterladen (CSV)",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
            )

        return

    # For non-DataFrame input, normalise to bytes
    content = None
    # file-like with read()
    if hasattr(data, 'read'):
        try:
            data.seek(0)
        except Exception:
            pass
        content = data.read()
    elif isinstance(data, (bytes, bytearray)):
        content = bytes(data)
    elif isinstance(data, str):
        content = data.encode('utf-8')
    else:
        # Fallback: stringify
        content = str(data).encode('utf-8')

    # Determine filename and mime
    fname = filename or "anonymized_data"
    mime = "application/octet-stream"

    # Use explicit file_type hint if provided
    if file_type:
        ft = file_type.lower()
        if ft in ('xlsx', 'xls', 'excel'):
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if not fname.lower().endswith(('.xlsx', '.xls')):
                fname = Path(fname).stem + '.xlsx'
        elif ft in ('csv',):
            mime = "text/csv"
            if not fname.lower().endswith('.csv'):
                fname = Path(fname).stem + '.csv'
        elif ft in ('txt', 'text'):
            mime = "text/plain"
            if not fname.lower().endswith('.txt'):
                fname = Path(fname).stem + '.txt'
    else:
        # Infer from filename extension if present
        lower = fname.lower()
        if lower.endswith(('.xlsx', '.xls')):
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif lower.endswith('.csv'):
            mime = "text/csv"
        elif lower.endswith('.txt'):
            mime = "text/plain"

    st.download_button(
        label="Anonymisierte Datei herunterladen",
        data=content,
        file_name=fname,
        mime=mime,
    )
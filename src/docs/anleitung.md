## Einführung

PII Toolbox ist eine benutzerfreundliche **Streamlit-App**, die es ermöglicht, Testdaten zu erzeugen und bestehende Daten im Tabellenformat zu pseudonymisieren, wobei die zu pseudonyisierenden Felder mit ähnlich klingenden Inhalten ersetzt werden. Ähnliche Funktionalitäten sind für unstrukturierte Texte vorgesehen, jedoch noch nicht implementiert.

## Menu
### Pseudonymizer-Konfiguration erstellen
Die Pseudonymisierung von ausgewählten Spalten einer Tabelle wird gesteuert über eine Konfigurationsdatei, in welcher die Regeln der Pseudonymisierung von Spalten mit Personneninformationen festegelegt sind. Diese Konfigurationsdatei kann mit diesem Menubefehl mittels Daten-Datei erzeugt werden. Die zu pseudonymisierende Daten-Datei. Wird dabei hochgeladen und eine json Datei erzeugt, welche einen leeren Eintrag für jede splate enthält. Anschliessend sind die Spalten mit Personeninhalten zu identifizieren und deren Konfiguration zu definieren.
``` json
{
    "student_id": {
        "pseudonymize": false,
        
    },
    "student_name": {
        "pseudonymize": false,
    },
}
```

### Testdaten-Konfiguration erstellen
todo
### Bestehende Datei pseudonymisieren

#### Funktionen

- Hochladen von **CSV**- oder **XLS/XLSX**-Dateien mit sensiblen Daten.
- Bereitstellung einer Konfigurationsdatei, um Pseudonymisierungsregeln für jede Spalte zu definieren.
- Interaktive Pseudonymisierung der Daten mit Überprüfungsmöglichkeiten.
- Herunterladen der pseudonymisierten Datei zur weiteren Nutzung.

---

## Funktionsweise

1. **Daten hochladen**:
   - Wähle eine **CSV**- oder **XLS/XLSX**-Datei mit den Daten aus, die pseudonymisiert werden sollen.

2. **Konfigurationsdatei hochladen**:
   - Stelle eine JSON-Konfigurationsdatei bereit, die folgendes spezifiziert:
     - Die Spalten, die pseudonymisiert werden sollen.
     - Die Pseudonymisierungsregeln für jede Spalte (z. B. Fake-Namen, E-Mail-Adressen).

3. **Pseudonymisierungsprozess**:
   - Die App wendet die in der Konfigurationsdatei definierten Regeln auf die Daten an.
   - Überprüfe und verifiziere die pseudonymisierten Ergebnisse direkt in der App.

4. **Ergebnis herunterladen**:
   - Lade die pseudonymisierte Datei zur sicheren Nutzung oder weiteren Verarbeitung herunter.

---

## Format der Konfigurationsdatei

Die Konfigurationsdatei ist eine JSON-Datei, in der jede zu pseudonymisierende Spalte definiert wird. Ein Beispiel ist unten dargestellt:

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
        "faker_function_input_parameters": {
            "unique_address_fields": ["adress", "postal_code"],
            "location_code_col": "plz",
            "location_data_col": "postal_code"
        }
    }
}
```


### Testdaten erstellen
todo!

### Dokumente pseudonymisieren
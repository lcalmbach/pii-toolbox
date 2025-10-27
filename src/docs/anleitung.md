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

## Pseudonymisierungsfunktionen
Die folgenden Funktionen sind implementiert und können in der Konfigurationsdatei als Wert für faker_function verwendet werden:

Pseudonymisierungsfunktionen
Die folgenden Funktionen sind implementiert und können in der Konfigurationsdatei als Wert für faker_function verwendet werden:

| Funktion         | Beschreibung                                                                                       | Parameter                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `random_number`  | Ersetzt jeden Wert in der Spalte durch eine Zufallszahl im Bereich von `min_value` bis `max_value`. | `{"min_value": 200000, "max_value": 700000, "unique": true}`                                |
| `random_address` | Ersetzt Adressen (Straße und Hausnummer) durch zufällige Adressen am selben Ort.                    | `{"unique_address_fields": ["addresse", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"}` |
| `blur_address`   | Ersetzt Adressen durch andere Adressen in derselben Straße, indem die Hausnummer geändert wird.    | `{"unique_address_fields": ["addresse", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"}` |
| `first_name`     | Ersetzt Vornamen durch zufällige Vornamen. Geschlechtsbasierte Namen können durch Angabe einer Geschlechtsspalte generiert werden. | `{"use_gender_col": true, "gender_col": "student_gender", "female": "2", "male": "1"}` |
| `last_name`      | Ersetzt Nachnamen durch zufällige Nachnamen. Geschlechtsbasierte Namen können durch Angabe einer Geschlechtsspalte sowie der codes für männlich/weiblich generiert werden. | `{"use_gender_col": true, "gender_col": "student_gender", "female": "2", "male": "1"}` |
| `shuffle_codes`      | Mischelt Codes. Es wird zuerst eine Liste der Codes aus der Quellspalte erstellt, die Zielspalte wird mit Zufallswerten aus dieser Liste gefüllt. Leere Zellen in der Quell-Spalte bleiben auch in der Zielspalte leer  | `{}` |
| `fill_with_random_distribution`      | Füllt die Zielspalte mit Zahlen aus einer Normalverteilung. Mittelwert und Standardabweichung werden der  Normalverteilung werden zuerst aus den Zahlen der Quellspalte berechnet. In der Zielspalte werden anschliessend aus dieser Verteilung Zufallszahlen generiert, deren Verteilung damit der Verteilung der Quellspalte ähnelt. Die Anzahl Dezimalstellen der Zielspalte kann mit dem Parameter *decimals* definiert werden.| `{"decimals": 2}` |


### Testdaten erstellen
todo!
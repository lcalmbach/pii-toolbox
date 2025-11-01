function_dict = {
  "incremental_int": {
    "description": "Generiert eine inkrementelle Liste von Integer-Werten",
    "parameters": ["min", "step"],
    "defaults": [0, 1],
    "help": [
        "Startwert der Serie", 
        "Schrittgrösse der Serie"],
    "example": """
"Feld1": {
      "data_type": "integer",
      "function": "incremental_int",
      "min": 1000,
      "step": 1
    }
"""
  },
  "list_int": {
    "description": "User gibt eine Liste von Integerzahlen vor aus welcher die Funktion zufällig wählt",
    "parameters": ["list"],
    "defaults": [10, 20, 30],
    "help": [
      "Komma separierte Liste von Integer Werten, aus denen die Applikation zufällig auswählt"
    ],
    "example": """
"Feld1": {
      "data_type": "integer",
      "function": "list_int",
      "list": [10, 20, 30]
    }
"""
  },
  "random_int": {
    "description": "Generiert eine Liste von Zufalls-Integer-Werten",
    "parameters": ["min", "max", "unique"],
    "defaults": [0, 1000, False],
    "help": [
      "Unterer Grenzwert",
      "Oberer Grenzwert",
      "Wenn True: nur eindeutige Werte (keine Duplikate)"
    ],
    "example": """
"Feld1": {
      "data_type": "integer",
      "function": "random_int",
      "min": 0,
      "max": 1000,
      "unique": false
    }
"""
  },
  "normal_float": {
    "description": "Generiert eine Liste von normalverteilten float Werten.",
    "parameters": ["min", "max", "mean", "std"],
    "defaults": [None, None, None,None],
    "help": [
        "Werte unterhalb dieser Limite werden neu generiert",
        "Werte oberhalb dieser Limite werden neu generiert",
        "Mittelwert der Verteilung",
        "Standardabweichung der Verteilung"
        ],
    "example": """
"Feld1": {
      "data_type": "float",
      "function": "normal_float",
      "min": 0.0,
      "max": 100.0,
      "mean": 50.0,
      "std": 10.0
    }
"""
  },
  "incremental_date": {
    "description": "Erzeugt eine Liste von inkrementellen Daten.",
    "parameters": ["start_date", "step_period", "step"],
    "defaults": ["2025-01-01", "day", 1],
    "help": ["Startdatum", "Zeiteinheit für Schritt (z.B. day, month)", "Schrittgrösse"],
    "example": """
"Feld1": {
      "data_type": "date",
      "function": "incremental_date",
      "start_date": "2025-01-01",
      "step_period": "day",
      "step": 1
    }
"""
  },
  "random_date": {
    "description": "Erzeugt eine Liste von zufälligen Daten innerhalb eines definierten Zeitintervalls.",
    "parameters": ["start_date", "end_date"],
    "defaults": ["2024-01-01", "2025-12-31"],
    "help": ["Startdatum des Intervalls", "Enddatum des Intervalls"],
    "example": """
"Feld1": {
      "data_type": "date",
      "function": "random_date",
      "start_date": "2024-01-01",
      "end_date": "2025-12-31"
    }
"""
  },
  "normal_date": {
    "description": "Erzeugt eine Liste von normalverteilten Daten mit gegebenen Mittelwert und Standardabweichung.",
    "parameters": ["average_date", "std_days"],
    "defaults": ["2024-01-01", 30],
    "help": ["Mittelpunkt des Zeitintervalls", "Standardabweichung in Tagen"],
    "example": """
"Feld1": {
      "data_type": "date",
      "function": "normal_date",
      "average_date": "2024-01-01",
      "std_days": 30
    }
"""},
  "date_add_random_days": {
    "description": "Verschiebt ein Datum zufällig um eine Anzahl Tage innerhalb eines Bereichs.",
    "parameters": ["lower", "upper"],
    "defaults": [-4, 4],
    "help": [
      "Untere Grenze des Zufallsintervalls in Tagen, um welche das Datum verschoben wird, z.B. -4",
      "Obere Grenze des Zufallsintervalls in Tagen, um welche das Datum verschoben wird, z.B. 4"
    ],
    "example": """
"Feld1": {
      "data_type": "date",
      "function": "date_add_random_days",
      "lower": -4,
      "upper": 4
    }
"""
  },
  "first_name": {
    "description": "Erzeugt zufällige Vornamen (optional geschlechtsabhängig).",
    "parameters": [
      "source",
      "gender_field",
      "percent_male",
      "gender_field_male_identfier"
    ],
    "defaults": ["ogd-bs", None, 50, "M"],
    "help": [
      "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
      "Feldname mit Geschlecht, falls vorhanden",
      "Anteil männlicher Vornamen bei Neugenerierung (Prozent)",
      "Wert, der Männer in der Geschlechtsspalte identifiziert"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "first_name",
      "source": "ogd-bs",
      "gender_field": "gender",
      "percent_male": 50,
      "gender_field_male_identfier": "M"
    }
"""
  },
  "last_name": {
    "description": "Erzeugt zufällige Nachnamen.",
    "parameters": ["source", "gender_field"],
    "defaults": ["ogd-bs", None],
    "help": [
      "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
      "Feldname mit Geschlecht, falls vorhanden"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "last_name",
      "source": "ogd-bs",
      "gender_field": null
    }
"""
  },
  "street_name": {
    "description": "Erzeugt zufällige Strassennamen.",
    "parameters": ["source", "location_field", "plz_field"],
    "defaults": ["ogd-bs", None, None],
    "help": [
      "'ogd-bs' für Strassennamen aus Basel, 'faker' für generische Namen.",
      "Feldname mit Gemeinde/Location, falls vorhanden",
      "Feldname mit PLZ, falls vorhanden"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "street_name",
      "source": "ogd-bs",
      "location_field": null,
      "plz_field": null
    }
"""
  },
  "housenumber": {
    "description": "Erzeugt zufällige Hausnummern.",
    "parameters": ["source", "location_field", "plz_field", "streetname_field"],
    "defaults": ["ogd-bs", None, None],
    "help": [
      "'ogd-bs' für Daten aus Basel, 'faker' für generische Daten.",
      "Feldname mit Gemeinde/Location, falls vorhanden",
      "Feldname mit PLZ, falls vorhanden",
      "Feldname mit Strassennamen, falls vorhanden"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "housenumber",
      "source": "ogd-bs",
      "location_field": null,
      "plz_field": null,
      "streetname_field": "str_name"
    }
"""
  },
  "streetname_housenumber": {
    "description": "Erzeugt kombiniert Strassennamen und Hausnummern als Paar.",
    "parameters": ["source", "location_field", "plz_field"],
    "defaults": ["ogd-bs", None, None],
    "help": [
      "'ogd-bs' für Daten aus Basel, 'faker' für generische Daten.",
      "Nur für source=ogd-bs: Wenn ein PLZ-Feld existiert, wähle Strassen passend zur PLZ (entweder location_field oder plz_field verwenden, nicht beides)",
      "Nur für source=ogd-bs: Wenn ein Gemeinde-Feld existiert, wähle Strassen für diese Gemeinde (entweder location_field oder plz_field verwenden, nicht beides)"
    ],
    "example": """
"Feld1": {
      "data_type": "array",
      "function": "streetname_housenumber",
      "source": "ogd-bs",
      "location_field": null,
      "plz_field": null
    }
"""
  },
  
  "location": {
    "description": "Erzeugt zufällige Ortsnamen oder weist Ort anhand PLZ zu, falls vorhanden.",
    "parameters": ["source", "plz_field"],
    "defaults": ["ogd-bs", None, None],
    "help": [
        "'ogd-bs' für Daten aus Basel, 'faker' für generische Daten.",
        "Falls vorhanden: Der Ort wird aus dem korrespondierenden PLZ-Wert in der gleichen Zeile zugewiesen"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "location",
      "source": "ogd-bs",
      "plz_field": "plz"
    }
"""
  },
  "plz4": {
    "description": "Wählt PLZ4-Werte aus einer definierten Auswahl aus.",
    "parameters": ["source", "cantons", "weights"],
    "defaults": ["ogd-bs", ["bs", "bl"], [5,1]],
    "help": [
        "'ogd-bs' für Daten aus Basel, 'faker' für generische Daten.",
        "Liste von Kantonen, aus denen PLZ4-Werte ausgewählt werden sollen",
        "Falls angegeben, muss für jeden Kanton ein Gewicht vorhanden sein; ein Gewicht von 2 bedeutet doppelte Wahrscheinlichkeit gegenüber 1"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "plz4",
      "source": "ogd-bs",
      "cantons": ["bs", "bl"],
      "weights": [5, 1]
    }
"""
  },
  "canton": {
    "description": "Wählt zufällige Kantone in der Schweiz. Für eine feste Auswahl nutzen Sie 'list_string', z.B. ['BS','BL'].",
    "parameters": ["field"],
    "defaults": ["kanton_name_kurz"],
    "help": ["kanton_name_kurz oder kanton_name"],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "canton",
      "field": "kanton_name_kurz"
    }
"""
  },
  "list_string": {
    "description": "User gibt eine Liste von Strings aus",
    "parameters": ["list"],
    "defaults": ["M", "F"],
    "weights": [1,1],
    "help": [
      "Komma separierte Liste von String-Werten, aus denen die Applikation zufällig auswählt"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "list_string",
      "list": ["M", "F"],
      "weights": [1, 1]
    }
"""
  },
  "address": {
    "description": "Erzeugt zufällige Adress-Komponenten (Strasse, Hausnummer, PLZ, Ort).",
    "parameters": ["fields", "labels"],
    "defaults": [["str_name", "housnumber"],["strasse", "hausnummer"]],
    "help": [
      "Diese Funktion liefert eine Liste von Feldern in beliebiger Kombination: ['streetname', 'housenumber', 'plz', 'location']",
      "Labels für die Feldnamen, z.B. ['strasse', 'hausnummer']"
    ],
    "example": """
"Feld1": {
      "data_type": "array",
      "function": "address",
      "fields": ["str_name", "housnumber"],
      "labels": ["strasse", "hausnummer"]
    }
"""
  },
  "ahv_nr": {
    "description": "Generiert eine Fake AHV Nummer nach dem Vorbild der echten AHV-Nummer (756.XXXX.XXXX.XX). Es kann nicht ausgeschlossen werden, dass diese Nummer auch als echte AHV-Nummer existiert",
    "parameters": [],
    "defaults": [],
    "help": [],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "ahv_nr"
    }
"""
  },
  "phone_nr": {
    "description": "Generiert eine Fake Telefon-Nummer, Es kann nicht ausgeschlossen werden, dass diese Nummer auch als echte AHV-Nummer existiert",
    "parameters": ["cc","ndc", "formats", "format_weights"],
    "defaults": [["+41"], ["61", "77", "78","79"], ["xx xxx xx xx"]],
    "help": [
      "Liste von Ländervorwahlen, z.B. ['+41']",
      "Liste von nationalen Netzvorwahlen, z.B. ['79','78','61']",
      "Liste von Nummernformaten",
      "Gewichte für die Formate; optional, gleiche Länge wie 'formats', gibt relative Häufigkeiten an"
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "phone_nr",
      "cc": ["+41"],
      "ndc": ["79"],
      "formats": ["xx xxx xx xx"],
      "format_weights": [1]
    }
"""
  },
  "email": {
    "description": "Generiert eine Fake Email-Adresse nach dem Muster Vorname.Nachname@provider. Die Provider müssen in einer Liste angegeben werden, ebenso das Vorname und Nachnamefeld. Wenn keine Namensfelder angegeben sind, so wird die faker Bibliothek verwendet.",
    "parameters": ["first_name_field","last_name_field", "providers"],
    "defaults": [None, None, ["gmail.com", "bluewin.ch", "gmx.ch"]],
    "help": [
      "Email wird generiert als Vorname.NachName@provider, gib das Feld an, in dem der Vorname steht",
      "Email wird generiert als Vorname.Nachname@provider, gib das Feld an, in dem der Nachname steht",
      "Liste von Providern, z.B. [gmail.com, bluewin.ch, gmx.ch]",
    ],
    "example": """
"Feld1": {
      "data_type": "string",
      "function": "email",
      "first_name_field": "Vorname",
      "last_name_field": "Nachname",
      "providers": ["gmail.com"],
    }
"""
  }
}

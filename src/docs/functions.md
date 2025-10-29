## Bestehende Datei anonymisieren

{
  "incremental_int": {
    "description": "Generiert eine inkrementelle Liste von Integer-Werten",
    "parameters": ["min", "step"],
    "defaults": [0, 1],
    "help": ["Startwert der Serie", "Schrittgrösse der Serie"]
  },
  "list_int": {
    "description": "User gibt eine Liste von Integerzahlen vor aus welcher die Funktion zufällig wählt",
    "parameters": ["list"],
    "defaults": [10, 20, 30],
    "help": [
      "Komma separierte Liste von Integer Werten, aus denen die Applikation zufällig auswählt"
    ]
  },
  "random_int": {
    "description": "Generiert eine Liste von Zufalls-Integer-Werten",
    "parameters": ["min", "max", "unique"],
    "defaults": [0, 1000, false],
    "help": [
      "Minimum Wert",
      "Maximum Wert"
    ]
  },
  "normal_float": {
    "description": "Generiert eine Liste von normalverteilten float Werten.",
    "parameters": ["min", "max", "mean", "std"],
    "defaults": [null, null, null,null],
    "help": [
        "Werte unterhalb dieser Limite werden neu generiert",
        "Werte oberhalb dieser Limite werden neu generiert",
        "Mittelwert der Verteilung",
        "Standardabweichung der Verteilung"
        ]
  },
  "incremental_date": {
    "parameters": ["start_date", "step_period", "step"],
    "defaults": ["2025-01-01", "day", 1],
    "help": []
  },
  "random_date": {
    "parameters": ["start_date", "end_date"],
    "defaults": ["2024-01-01", "2025-12-31"],
    "help": []
  },
  "date_add_random_days": {
    "parameters": ["lower", "upper"],
    "defaults": [4, 4],
    "help": [
      "Untere Grenze des Zufallsintervalls, z.B. -4",
      "Obere Grenze des Zufallsintervalls, z.B. 4"
    ]
  },
  "first_name": {
    "parameters": [
      "source",
      "gender_field",
      "percent_male",
      "gender_field_male_identfier"
    ],
    "defaults": ["ogd-bs", null, 50, "M"],
    "help": [
      "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
      "Falls Parameter gesetzt (nur bei Daten maskieren): Wähle männliche, weiblliche vornamen abhängig vom Geschlecht in Spalte X",
      "Bei neu Generierung kann ein spezifizierter Anteil männliche vornamen generiert werden",
      "Bei Ableitung aus Geschlechtsspalte gibt den Code an, der Männer identifiziert."
    ]
  },
  "last_name": {
    "parameters": ["source", "gender_field"],
    "defaults": ["ogd-bs", null],
    "help": ["'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker."]
  },
  "street_name": {
    "parameters": ["source", "location_field", "plz_field"],
    "defaults": ["ogd-bs", null, null],
    "help": ["'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker."]
  },
  "housenumber": {
    "parameters": ["source", "location_field", "plz_field", "streetname_field"],
    "defaults": ["ogd-bs", null, null],
    "help": ["'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker."]
  },
  "streetname_housenumber": {
    "parameters": ["source", "location_field", "plz_field"],
    "defaults": ["ogd-bs", null, null],
    "help": [
      "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
      "Nur für source=ogd-bs: Wenn PLZ Feld existiert soll Strasse für diese PLZ gewählt werden, entwder location_field oder plz_field als Abhängigkeits-Feld angeben, nciht beide zusammen",
      "Nur für source=ogd-bs: Wenn Location (Gemeinde) Feld existiert soll Strasse für diesen Gemeinde gewählt werden, entwder location_field oder plz_field als Abhängigkeits-Feld angeben, nciht beide zusammen"
    ]
  },
  
  "location": {
    "parameters": ["source", "plz_field"],
    "defaults": ["ogd-bs", null, null],
    "help": [
        "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
        "if present, the location will be assigned from the correcpsonding PLZ value in the same row"
    ]
  },
  "plz4": {
    "description": "Selects ",
    "parameters": ["source", "cantons", "weights"],
    "defaults": ["ogd-bs", ["bs", "bl"], [5,1]],
    "help": [
        "'ogd-bs' für Namen aus Basel, 'faker' für Namen aus Python Faker.",
        "list of cantons from where to select plz4 values",
        "if present, it must be filled with 1 value for each canton, a value with 2 will be selected 2 times more often than a value of 1."
    ]
  },
  "canton": {
    "description": "Selects random cantons in Switzerland. If you need a selection of cantons, you may use the function 'list_string' e.g. ['BS', 'BL'] ",
    "parameters": ["field"],
    "defaults": ["kanton_name_kurz"],
    "help": ["kanton_name_kurz oder kanton_name"]
  },
  "list_string": {
    "description": "User gibt eine Liste von Strings aus",
    "parameters": ["list"],
    "defaults": ["M", "F"],
    "weights": [1,1],
    "help": [
      "Komma separierte Liste von Strings Werten, aus denen die Applikation zufällig auswählt"
    ]
  },
  "address": {
    "description": "Selects random cantons in Switzerland. If you need a selection of cantons, you may use the function 'list_string' e.g. ['BS', 'BL'] ",
    "parameters": ["fields", "labels"],
    "defaults": [["str_name", "housnumber"],["strasse", "hausnummer"]],
    "help": ["This function returns a list of arrays, any combination of the fields: ['streetname', 'housenamuber', 'plz', 'location']"]
  },
  "ahv_nr": {
    "description": "Generiert eine Fake AHV Nummer, Es kann nicht ausgeschlossen werden, dass diese Nummer auch als echte AHV-Nummer existiert",
    "parameters": [],
    "defaults": [],
    "help": []
  },
  "phone_nr": {
    "description": "Generiert eine Fake Telefon-Nummer, Es kann nicht ausgeschlossen werden, dass diese Nummer auch als echte AHV-Nummer existiert",
    "parameters": ["cc","ndc", "formats", "format_weights"],
    "defaults": [["+41"], ["61", "77", "78","79"], ["xx xxx xx xx"]],
    "help": [
      "List of country codes e.g. [+41]",
      "list of national destination e.g. [79, 78, 61]",
      "list of formats"
    ]
  },
  "email": {
    "description": "Generiert eine Fake Email-Adresse",
    "parameters": ["first_name_field","last_name_field", "providers", "source"],
    "defaults": [null, null, ["gmail.com", "bluewin.ch", "gmx.ch"]],
    "help": [
      "Email wird generiert als Vorname.NachName@provider, gib das Feld an in dem der Vorname steht",
      "Email wird generiert als Vorname.Nachname@provider, gib das Feld an in dem der Vorname steht",
      "Liste von Provider, z.b. [gmail.com, bluewin.ch, gmx.ch]",
      "Wenn es keine Vor/Nachname Spalten gibt, dann erzeuge diese Info entweder mit ogd-bs oder mit faker"
    ]
  }
}

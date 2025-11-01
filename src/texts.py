app_info = """Die App PII-Toolbox ermöglicht die Anonymisierung strukturierter und unstrukturierter Daten und stellt damit eine sichere und datenschutzkonforme Lösung zur Verfügung, um realistische Testdaten für die Applikationsentwicklung und Modellierung zu erzeugen. Darüber hinaus können Testdaten auch unabhängig von bestehenden „scharfen“ Daten vollständig neu generiert werden. Für die Erstellung von Namen und Adressen lassen sich öffentliche Datensätze des Kantons Basel-Stadt nutzen, um möglichst realitätsnahe Testdaten bereitzustellen.

### Vorteile anonymisierter Testdaten  

- **Einhaltung von Datenschutzbestimmungen**  
  Durch die Anonymisierung können sensible Informationen anonymisiert werden, sodass sie für Testzwecke genutzt werden können, ohne gegen Datenschutzrichtlinien zu verstossen.  

- **Realistische Testbedingungen**  
  Anonymisierte Daten behalten die Struktur und Eigenschaften der Originaldaten bei. Dadurch können Anwendungen unter realitätsnahen Bedingungen entwickelt und getestet werden.  

- **Verbesserung der Softwarequalität**  
  Entwickler können ihre Anwendungen mit echten, aber nicht personenbezogenen Daten testen. Dies hilft, Fehler frühzeitig zu erkennen und die Stabilität der Software zu verbessern.  

- **Optimierung von Modellen**  
  Für Machine-Learning-Modelle oder statistische Analysen bietet die Anonymisierung die Möglichkeit, mit repräsentativen, aber sicheren Daten zu arbeiten – ohne auf synthetische oder zufällig generierte Werte zurückgreifen zu müssen.  

- **Erweiterte Nutzungsmöglichkeiten für Dokumente**  
  Dokumente, die persönliche Informationen wie Namen oder Telefonnummern enthalten, dürfen oft nur einem sehr kleinen Personenkreis zugänglich gemacht werden. Allerdings könnten diese Dokumente auch für eine breitere Zielgruppe von Interesse sein, insbesondere für Personen, die sich für den Inhalt, nicht aber für die personenbezogenen Daten interessieren. Durch die Entfernung oder Anonymisierung sensibler Daten können solche Dokumente für eine grössere Nutzergruppe zugänglich gemacht oder auch für das **Fine-Tuning von verwaltungsinternen Modellen oder andere analytische Zwecke** verwendet werden.  

### Transparenz und bessere Nutzung der Software  

Ein weiterer grosser Vorteil dieser App ist, dass wir **unseren Code gemeinsam mit den anaonymisierten Testdaten veröffentlichen können**. Dadurch haben Nutzer die Möglichkeit:  

- Die Software besser zu verstehen und nachzuvollziehen.  
- Eigene Tests und Anpassungen vorzunehmen, ohne Zugriff auf produktive Daten zu benötigen.  
- Die Qualität und Funktionalität der Applikation eigenständig zu prüfen.  

Durch diese offene und transparente Vorgehensweise wird die Nutzung unserer Software erleichtert und der Entwicklungsprozess für alle Beteiligten sicherer und effizienter gestaltet.  
"""

info_anonymizer = """Mit dieser Option wird eine bestehende Tabelle (XLSX oder CSV) anhand einer Steuerungsdatei im JSON‑Format anonymisiert. Es werden nur die Spalten verändert, für die in der Steuerungsdatei eine Anonymisierungsfunktion angegeben ist.

Vorgehen:
- Ziehe die zu anonymisierende Datei in das obere Feld (oder wähle sie aus).
- Ziehe die Steuerungsdatei (JSON) in das untere Feld.
- Die Steuerungsdatei muss für jede zu anonymisierende Spalte eine Funktion definieren (siehe Anleitung).
- Drücke auf „Anonymisierung starten“. Nach Abschluss kannst du die anonymisierte Datei herunterladen."""

konfig_erstellen = """Diese Option erstellt aus einer Beispieldatei (XLSX oder CSV) automatisch eine Steuerungsdatei im JSON‑Format, die als Vorlage für die Anonymisierung dient.

Vorgehen:
- Lade eine Beispieldatei hoch. Die Datei muss mindestens eine Kopfzeile mit Spaltennamen enthalten.
- PII Toolbox erstellt daraus eine JSON‑Konfiguration mit Vorschlägen zu Funktionen für die einzelnen Spalten.
- Prüfe und passe die erzeugte JSON‑Datei manuell an: Trage für jede Spalte mit personenbezogenen Daten (z. B. Vorname, Nachname, Telefonnummer, AHV, Adresse) die gewünschte Anonymisierungsfunktion ein.
- Verwende die Anleitung für Details zu möglichen Funktionen und Parametern."""

testdaten_erstellen = """Mit dieser Option können Testdaten erzeugt werden, ohne eine Ausgangsdatei hochzuladen. Eine Konfigurationsdatei (JSON) legt die Anzahl der Datensätze, das Ausgabeformat, Encoding, Dateiname, Separator sowie die Spalten und deren Erzeugungsregeln fest.

Vorgehen:
- Lade die Konfigurationsdatei im JSON‑Format hoch (oder aktiviere den Demo‑Modus).
- Passe gegebenenfalls Anzahl der Datensätze und Dateiname an.
- Starte die Generierung. Danach kannst du die erzeugte Beispieldatei herunterladen."""

anonymize_texts = """Diese Option erkennt und maskiert personenbezogene Angaben in Textdokumenten (z. B. Namen, Adressen, Telefonnummern), die auf Personen schliessen lassen.

Modi:
- Interaktive Eingabe: Text direkt eingeben oder einfügen.
- Dokument hochladen: PDF, TXT, CSV oder TAB-Datei hochladen; der Text wird extrahiert und angezeigt.
- Demo: Beispieldokument verwenden.

Hinweise:
- Nach dem Extrahieren wird der Text angezeigt. Klicke auf "Text Anonymisieren", um die Ersetzung vorzunehmen. Persönliche Angaben werden durch Platzhalter ersetzt.
- Bei PDFs kann die Texterkennung von der Qualität der PDF‑Datei abhängen.
- Bei Tabellen (CSV/TAB) wird die Tabelle zeilenweise in einen lesbaren Text umgewandelt vor der Anonymisierung."""
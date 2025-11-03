import logging
from functools import lru_cache
from typing import Optional, Dict, Any, Iterable, Literal

import pandas as pd
import numpy as np
import re

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configure logger
logger = logging.getLogger(__name__)

# Swiss phone regex (allows +41 / 0041 or leading 0 prefixes)
ch_phone_regex = r"(?:(?:\+41|0041)\s?\(?\d{1,2}\)?|0\d{2})[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
ch_phone_pattern = Pattern(name="ch_phone_full", regex=ch_phone_regex, score=0.85)

ch_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ch_phone_pattern],
    supported_language="de",
)

LOWER_PARTICLES = {"von", "van", "de", "der", "den", "di", "du", "le", "la"}
DENY_START = {"Der", "Die", "Das", "Dem", "Den", "Ein", "Eine", "Einem", "Einen"}
ALLCAPS_RE = re.compile(r"^[A-ZÄÖÜ]{2,}$")

def _is_plausible_person(span_text: str) -> bool:
    """
    Strenge Heuristiken für deutsche Personennamen:
    - mind. 2 Teile
    - keine Ziffern, keine reinen ALL-CAPS Teile
    - Teile sind Title-Case (Ausnahme: Partikel wie 'von', 'de' in Kleinbuchstaben)
    - kein verbotener Satzanfang (Der/Die/Das/…)
    """
    if not span_text or any(ch.isdigit() for ch in span_text):
        return False

    parts = [p for p in span_text.strip().split() if p]
    if len(parts) < 2 or parts[0] in DENY_START:
        return False

    # alle Teile prüfen
    ok = 0
    for i, p in enumerate(parts):
        if ALLCAPS_RE.match(p):  # CHF, AG, GmbH, etc.
            return False
        if p.lower() in LOWER_PARTICLES:
            # Partikel erlaubt, zählt aber nicht als Namens-Token
            continue
        # Normaler Name: Title-Case (erstes gross, rest klein)
        if p[0].isupper() and p[1:].islower():
            ok += 1
        else:
            return False

    # mindestens zwei echte Namens-Token (ohne Partikel)
    return ok >= 2

@lru_cache(maxsize=1)
def get_analyzer_anonymizer():
    """
    Create and return (analyzer, anonymizer), cached.
    Tries spaCy DE; falls back to predefined EN + minimal DE recognizers to keep 'de' supported.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_md"}],
    }

    registry = RecognizerRegistry()

    def _ensure_de_supported(reg: RecognizerRegistry):
        """
        Stellt sicher, dass 'de' in reg.supported_languages enthalten ist.
        Falls kein DE-Recognizer vorhanden, fügt einen minimalistischen PERSON-PatternRecognizer hinzu.
        """
        langs = set(reg.supported_languages or [])
        if "de" not in langs:
            # sehr einfacher PERSON-Regex (Vor- + Nachname, großgeschrieben), besser als nichts
            person_pattern = Pattern(
                name="de_person_simple",
                regex=r"\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+\b",
                score=0.40,
            )
            de_person = PatternRecognizer(
                supported_entity="PERSON",
                patterns=[person_pattern],
                supported_language="de",
            )
            reg.add_recognizer(de_person)

    try:
        # --- Primärweg: deutsches spaCy Modell ---
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        registry.load_predefined_recognizers()  # lädt v.a. EN
        # spaCy-Recognizer auf PERSON & de begrenzen
        try:
            registry.add_recognizer(
                SpacyRecognizer(supported_language="de", supported_entities=["PERSON"])
            )
        except Exception as ex:
            logger.warning("Spacy recognizer could not be added: %s", ex)

        # custom CH-Phone
        registry.add_recognizer(ch_phone_recognizer)

        # sicherstellen, dass 'de' wirklich drin ist
        _ensure_de_supported(registry)

        # Sprachen konsistent halten
        langs = sorted(set(registry.supported_languages or []))
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=langs)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer

    except Exception as e:
        # --- Fallback ohne spaCy: trotzdem 'de' sicherstellen ---
        logger.warning("Failed to init spaCy DE, falling back: %s", e)

        try:
            registry.load_predefined_recognizers()  # EN
        except Exception:
            logger.exception("Failed to load predefined recognizers")

        try:
            registry.add_recognizer(ch_phone_recognizer)  # DE (PHONE_NUMBER)
        except Exception:
            logger.exception("Failed to add custom phone recognizer")

        # minimalen DE PERSON-Recognizer hinzufügen, wenn nötig
        _ensure_de_supported(registry)

        langs = sorted(set(registry.supported_languages or []))
        analyzer = AnalyzerEngine(registry=registry, supported_languages=langs)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer


class TextAnonymizer:
    """Wrapper around Presidio analyzer + anonymizer for simple usage."""

    def __init__(self, operators: Optional[Dict[str, OperatorConfig]] = None, *, entities: Optional[list[str]] = None):
        # Standard: nur PERSON anonymisieren
        self.entities = entities or ["PERSON"]
        self.analyzer, self.anonymizer = get_analyzer_anonymizer()

        # Default-Operatoren
        self.operators = operators or {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            # PHONE_NUMBER wirkt nur, wenn du self.entities entsprechend erweiterst
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 10, "from_end": True}),
        }

        # Optionale Quelle für Tabelleneingaben (falls du anonymize_table ohne table-Arg nutzt)
        self.input = None  # kannst du extern befüllen: self.input = list[dict] / DataFrame / etc.

    def anonymize_text(self, text: str) -> str:
        """
        Analyze text and anonymize detected entities.
        Returns anonymized text; on errors or when nothing found returns original text.
        """
        if text is None:
            return text

        try:
            results = self.analyzer.analyze(
                text=text,
                language="de",
                entities=self.entities,      # typischerweise ["PERSON"]
                score_threshold=0.70          # wichtig: zu "gierige" Treffer rausfiltern
            )
        except Exception as e:
            logger.exception("Analyzer failed: %s", e)
            return text

        if not results:
            return text

        # --- STRIKTE NACHFILTERUNG NUR AUF PLAUSIBLE NAMEN ---
        filtered = []
        for r in results:
            if r.entity_type != "PERSON":
                # wenn du NUR PERSON willst, alles andere verwerfen
                continue
            span = text[r.start:r.end]
            if _is_plausible_person(span):
                filtered.append(r)

        if not filtered:
            return text

        try:
            anon = self.anonymizer.anonymize(
                text=text,
                analyzer_results=filtered,
                operators=self.operators,
            )
            return getattr(anon, "text", str(anon))
        except Exception as e:
            logger.exception("Anonymization failed: %s", e)
            return text

    def anonymize_table(
        self,
        table: Optional[Iterable[dict]] = None,
        *,
        keep_types: bool = True,
        return_format: Literal["dataframe", "records"] = "dataframe",
        columns: Optional[list[str]] = None,
        errors: Literal["raise", "log", "ignore"] = "log",
    ):
        """
        Anonymisiert jede Zelle einer Tabelle mittels self.anonymize_text.
        - keep_types=True: Nicht-String-Zellen bleiben unverändert.
          False: Nicht-Strings werden zu String gecastet und anonymisiert.
        - return_format: 'dataframe' (default) oder 'records' (List[dict])
        - columns: Nur diese Spalten anonymisieren (None = alle)
        - errors: 'raise' = Exception durchreichen,
                  'log'   = loggen und Originalwert zurückgeben,
                  'ignore'= stillschweigend Originalwert zurückgeben
        """
        source = table if table is not None else self.input
        df = pd.DataFrame(source).copy()

        target_cols = columns if columns is not None else list(df.columns)

        def _handle_error(e: Exception, original: Any):
            if errors == "raise":
                raise
            if errors == "log":
                try:
                    logger.exception("Anonymization failed for value %r: %s", original, e)
                except NameError:
                    print(f"[anonymize_table] failed for {original!r}: {e}")
            return original  # 'log' oder 'ignore'

        def _anon_cell(x: Any):
            # NaN/None: unverändert lassen
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return x

            if isinstance(x, str):
                try:
                    return self.anonymize_text(x)
                except Exception as e:
                    return _handle_error(e, x)

            if keep_types:
                return x
            else:
                try:
                    return self.anonymize_text(str(x))
                except Exception as e:
                    return _handle_error(e, x)

        for c in target_cols:
            if c in df.columns:
                df[c] = df[c].map(_anon_cell)

        if return_format == "records":
            return df.to_dict(orient="records")
        return df

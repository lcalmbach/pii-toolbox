import logging
from functools import lru_cache
from typing import Optional, Dict

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

@lru_cache(maxsize=1)
def get_analyzer_anonymizer():
    """Create and return a tuple (analyzer, anonymizer).

    This is cached to avoid re-creating heavy NLP models for each request.
    Falls back to predefined recognizers if spaCy model is not available.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "de", "model_name": "de_core_news_md"},
        ],
    }

    registry = RecognizerRegistry()
    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        registry.load_predefined_recognizers()
        # Add spaCy-based recognizer; if the model is not installed this may raise
        try:
            registry.add_recognizer(SpacyRecognizer(supported_language="de"))
        except Exception as ex:
            logger.warning("Spacy recognizer could not be added: %s", ex)
        # Add custom swiss phone recognizer
        registry.add_recognizer(ch_phone_recognizer)
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception as e:
        # Fallback: use predefined recognizers only
        logger.warning("Failed to initialize spaCy NLP engine, falling back to predefined recognizers: %s", e)
        try:
            registry.load_predefined_recognizers()
        except Exception:
            logger.exception("Failed to load predefined recognizers")
        try:
            registry.add_recognizer(ch_phone_recognizer)
        except Exception:
            logger.exception("Failed to add custom phone recognizer")
        analyzer = AnalyzerEngine(registry=registry)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer


class TextAnonymizer:
    """Wrapper around Presidio analyzer + anonymizer for simple usage.

    Example:
        ta = TextAnonymizer(text)
        result = ta.anonymize()
    """

    def __init__(self, text: str, operators: Optional[Dict[str, OperatorConfig]] = None):
        self.text = text or ""
        self.analyzer, self.anonymizer = get_analyzer_anonymizer()
        # Default operators: replace everything by placeholders, mask phone numbers
        self.operators = operators or {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 10, "from_end": True}),
        }

    def anonymize(self) -> str:
        """Analyze the text and anonymize detected entities.

        Returns anonymized text as string. If analysis fails or no entities found,
        returns the original text.
        """
        try:
            analyzer_results = self.analyzer.analyze(text=self.text, language="de")
        except Exception as e:
            logger.exception("Analyzer failed: %s", e)
            return self.text

        if not analyzer_results:
            # nothing to anonymize
            return self.text

        try:
            anonymized_result = self.anonymizer.anonymize(
                text=self.text,
                analyzer_results=analyzer_results,
                operators=self.operators,
            )
            return getattr(anonymized_result, 'text', str(anonymized_result))
        except Exception as e:
            logger.exception("Anonymization failed: %s", e)
            return self.text

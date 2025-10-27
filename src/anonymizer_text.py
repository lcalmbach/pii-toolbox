from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# https://deepwiki.com/microsoft/presidio/4-usage-guides

ch_phone_regex = r"(?:(?:\+41|0041)\s?\(?\d{1,2}\)?|0\d{2})[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
ch_phone_pattern = Pattern(name="ch_phone_full", regex=ch_phone_regex, score=0.85)

ch_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ch_phone_pattern],
    supported_language="de",
)

class TextAnonymizer:
    def __init__(self, text):
        self.text = text

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "de", "model_name": "de_core_news_md"},
            ],
        }

        # Create NLP engine
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Create registry with built-ins + spaCy NER for German
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(SpacyRecognizer(supported_language="de"))
        registry.add_recognizer(ch_phone_recognizer)
        # Analyzer + Anonymizer
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
        self.anonymizer = AnonymizerEngine()

    def anonymize(self):
        analyzer_results = self.analyzer.analyze(text=self.text, language="de")

        anonymized_result = self.anonymizer.anonymize(
            text=self.text,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}),
                "PERSON":  OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 10, "from_end": True}),
            },
        )
        # Return a string (the anonymized text), not the result object
        return anonymized_result.text

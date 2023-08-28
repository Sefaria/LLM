from typing import List, Any
from langchain.llms import OpenAI
from langchain import PromptTemplate
from constants import GPT_COMPLETION_END_INDICATOR, GPT_PROMPT_END_INDICATOR
from langchain.schema import BaseOutputParser
from dataclasses import dataclass
from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer
from create_citation_input_for_fine_tuning import GptEntityClassificationTrainingGenerator, SPAN_LABEL_TO_CLASSICATION_TAG
import re


entity_recognizer_model = "curie:ft-sefaria:en-ner-2023-08-06-12-59-47"
entity_classifier_model = "ada:ft-sefaria:en-entity-classification-2023-08-10-18-23-01"


@dataclass
class Entity:
    text: str
    start: int
    end: int
    label: str = None


@dataclass
class EntityDoc:
    text: str
    entities: List[Entity]

    def validate(self, original_text):
        if original_text != self.text:
            raise Exception(f"Original text is not equal to text.\nOrignal:\n{original_text}\nFinal:\n{self.text}")

        for ent in self.entities:
            if self.text[ent.start:ent.end] != ent.text:
                raise Exception(f"Entity {ent} does not match text {self.text[ent.start:ent.end]}")

    def __str__(self):
        entities_str = '\n'.join(e.__repr__() for e in self.entities)
        return f"TEXT:\n{self.text}\nENTITIES:\n{entities_str}"


pre_wrapper = "{{"
post_wrapper = "}}"
pattern = re.compile(fr'{re.escape(pre_wrapper)}(.+?){re.escape(post_wrapper)}')
bracket_normalizer = RegexNormalizer(fr'{re.escape(pre_wrapper)}|{re.escape(post_wrapper)}', r'')
strip_normalizer = RegexNormalizer(r'^\s*|\s*$', r'')
normalizer = NormalizerComposer(steps=[bracket_normalizer, strip_normalizer])


class EntityParser(BaseOutputParser[EntityDoc]):

    @property
    def _type(self) -> str:
        return self.__class__.__name__

    def get_format_instructions(self) -> str:
        return "Wrap entities with double curly brackets."

    def parse(self, text: str) -> EntityDoc:
        entities = []
        for entity_match in pattern.finditer(text):
            entities += [self._create_entity(entity_match)]
        cleaned_text = normalizer.normalize(text)
        corrected_entities = self._correct_entity_locations(text, entities)
        return EntityDoc(text=cleaned_text, entities=corrected_entities)

    def _correct_entity_locations(self, text, entities) -> List[Entity]:
        mapping = normalizer.get_mapping_after_normalization(text, reverse=True)
        orig_indices = [(e.start, e.end) for e in entities]
        new_indices = normalizer.convert_normalized_indices_to_unnormalized_indices(orig_indices, mapping, reverse=True)
        for e, (start, end) in zip(entities, new_indices):
            e.start = start
            e.end = end
        return entities

    @staticmethod
    def _create_entity(entity_match: re.Match) -> Entity:
        start = entity_match.start(1)
        end = entity_match.end(1)
        entity_text = entity_match.group(1)
        return Entity(entity_text, start, end)


class EntityTagger:

    def __init__(self):
        self._llm_recognizer = OpenAI(model=entity_recognizer_model)
        self._llm_classifier = OpenAI(model=entity_classifier_model)
        self._parser = EntityParser()
        self._template = PromptTemplate.from_template("{input}" + GPT_PROMPT_END_INDICATOR)

    def predict(self, text) -> EntityDoc:
        doc = self._recognize_entities(text)
        doc = self._classify_entities(doc)
        return doc

    def _recognize_entities(self, text):
        prompt = self._template.format(input=text)
        output = self._llm_recognizer(prompt, stop=[GPT_COMPLETION_END_INDICATOR])
        doc = self._parser.parse(output)
        doc.validate(text)
        return doc

    def _classify_entities(self, doc: EntityDoc) -> EntityDoc:
        for entity in doc.entities:
            entity.label = self._classify_entity(doc.text, entity)
        return doc

    def _classify_entity(self, text, entity: Entity) -> str:
        generator = GptEntityClassificationTrainingGenerator()
        prompt = generator.create_prompt(text, entity.start, entity.end)
        output = self._llm_classifier(prompt, stop=[GPT_COMPLETION_END_INDICATOR])
        output = output.strip()
        if output not in SPAN_LABEL_TO_CLASSICATION_TAG.values():
            print(f"NOT good '{output}'")
            raise AssertionError
        return output


if __name__ == '__main__':
    tagger = EntityTagger()
    text = "The Torah says in Genesis 1:6 'Thou shalt be cool'. The Ben Yehoyda says that this means you should learn Torah all day."
    doc = tagger.predict(text)
    print(doc)



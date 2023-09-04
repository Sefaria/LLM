from tqdm import tqdm
from typing import List, Any
from langchain.llms import OpenAI
from langchain import PromptTemplate
from linker.fine_tune.project_scripts.constants import GPT_COMPLETION_END_INDICATOR, GPT_PROMPT_END_INDICATOR
from langchain.schema import BaseOutputParser
from dataclasses import dataclass
from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer, AbstractNormalizer
from linker.fine_tune.project_scripts.create_citation_input_for_fine_tuning import GptEntityClassificationTrainingGenerator, SPAN_LABEL_TO_CLASSICATION_TAG
import re
import random
from util.general import load_mongo_docs, get_removal_list
from util.sentencizer import sentencize
from db_manager import MongoProdigyDBManager

import langchain
from langchain.cache import SQLiteCache
from sefaria.spacy_function_registry import inner_punct_tokenizer_factory
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English


langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

random.seed(613)

entity_recognizer_model = "curie:ft-sefaria:en-ner-2023-08-30-12-45-23"  # "curie:ft-sefaria:en-ner-2023-08-06-12-59-47"
entity_classifier_model = "ada:ft-sefaria:en-ner-2023-08-30-11-17-37"  # "ada:ft-sefaria:en-entity-classification-2023-08-10-18-23-01"

nlp = English()
nlp.tokenizer = inner_punct_tokenizer_factory()(nlp)


class ExampleGenerator:

    collections = ["ner_en_output2_gold_left"]  # ["ner_en_sent_input", "ner_en_sent_web_input"]

    def __init__(self):
        docs_list = []
        for collection in self.collections:
            docs = list(filter(self._filter_bad_sentences, load_mongo_docs(collection)))
            random.shuffle(docs)
            docs_list += [docs]
        min_count = min(len(d) for d in docs_list)
        self._data = []
        for docs in docs_list:
            self._data += docs[:min_count]
        random.shuffle(self._data)

    @staticmethod
    def _filter_bad_sentences(doc):
        text = doc['text']
        if "&lt;" in text or "&gt;" in text:
            return False
        return True

    def get(self):
        return self._data


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
    meta: dict = None

    def validate(self, original_text):
        if original_text != self.text:
            realign_entities(original_text, self)
            # raise Exception(f"Original text is not equal to text.\nOrignal:\n{original_text}\nFinal:\n{self.text}")

        for ent in self.entities:
            if self.text[ent.start:ent.end] != ent.text:
                raise AssertionError(f"Entity {ent} does not match text '{self.text[ent.start:ent.end]}'")

    def spacy_serialize(self) -> dict:
        spacy_doc = nlp(self.text)
        return {
            "text": self.text,
            "meta": self.meta,
            "spans": [self._spacy_serialize_entity(spacy_doc, entity) for entity in self.entities]
        }

    @staticmethod
    def _spacy_serialize_entity(spacy_doc: Doc, entity: Entity) -> dict:
        from sefaria.model.linker.ref_part import span_inds
        span = spacy_doc.char_span(entity.start, entity.end)
        span_start, span_end = span_inds(span)
        return {
            "start": entity.start,
            "end": entity.end,
            "token_start": span_start,
            "token_end": span_end-1,
            "label": entity.label,
        }

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
        self._llm_recognizer = OpenAI(model=entity_recognizer_model, temperature=0)
        self._llm_classifier = OpenAI(model=entity_classifier_model, temperature=0)
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


def realign_entities(original_text: str, doc: EntityDoc) -> EntityDoc:
    removal_list = get_removal_list(original_text, doc.text)
    temp_normalizer = AbstractNormalizer()
    mapping = temp_normalizer.get_mapping_after_normalization(original_text, removal_list, reverse=False)
    old_inds = [(entity.start, entity.end) for entity in doc.entities]
    new_inds = temp_normalizer.convert_normalized_indices_to_unnormalized_indices(old_inds, mapping, reverse=False)
    for (new_start, new_end), entity in zip(new_inds, doc.entities):
        entity.start = new_start
        entity.end = new_end
    doc.text = original_text
    return doc


if __name__ == '__main__':
    tagger = EntityTagger()
    # text = "See also Rambam, Commentary on the Mishneh , Avot 5:6; idem , Guide to the Perplexed , Part II, chaps. 25 and 29;"
    # doc = tagger.predict(text)
    # print(doc)

    my_db = MongoProdigyDBManager("ner_en_gpt_output2")
    my_db.output_collection.delete_many({})
    generator = ExampleGenerator()
    for d in tqdm(generator.get()):
        text = d['text']
        for sent in sentencize(text):
            try:
                doc = tagger.predict(sent)
                doc.meta = d['meta']
                print(doc)
                mongo_doc = doc.spacy_serialize()
                my_db.output_collection.insert_one(mongo_doc)
            except (AssertionError, AttributeError) as e:
                print("ERROR", sent)
            print()



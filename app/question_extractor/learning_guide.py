import django
django.setup()
from sefaria.model import *
import datetime, time, os, pickle
from typing import List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory


ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
claude_opus_llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229", max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
claude_sonnet_llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229", max_tokens=4096, anthropic_api_key=ANTHROPIC_API_KEY)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    safety_settings={
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    },
    temperature=.7,
    max_output_tokens=8096,
    google_api_key=os.environ.get('GOOGLE_API_KEY'))


# Models used in analyze_sources
class Commentary(BaseModel):
    ref: str = Field(description="Identifier for the commentary text")
    index_title: str = Field(description="Title of the book")
    author: str = Field(description="Author of the commentary")
    en_text: str = Field(description="English translation of the commentary")


class BaseText(BaseModel):
    ref: str = Field(description="Identifier for the text")
    english_version: str = Field(description=" English translation of the text")
    hebrew_version: str = Field(description="Original Hebrew version of the text")
    num_linked: int = Field(description="Number of linked commentaries")
    linked_commentaries: List[Commentary] = Field(description="Commentaries linked to this text")

    @classmethod
    def from_ref(cls, r: Ref, exclude_refs = None):
        """

        :param r: Ref
        :param exclude_refs: list of Refs to exclude from the linked_commentaries.  Can be Index level or lower.
        :return: baseText and linkedRefs
        """
        # Get the English text for this ref
        tc = r.text('en')
        en_text = tc.remove_html_and_make_presentable(tc.text)
        he_text = tc.remove_html_and_make_presentable(r.text("he").text)

        # get all the connected commentary that is in English
        linked_refs = r.linkset().refs_from(r)
        eng_linked_refs = [a for a in filter(lambda x: x.is_text_fully_available('en'), linked_refs)]
        if exclude_refs:
            eng_linked_refs = [a for a in eng_linked_refs if not any(e.contains(a) for e in exclude_refs)]
        commentaries = []
        for lr in eng_linked_refs:
            c = {
                "ref": lr.normal(),
                "index_title": lr.index.title,
                "author": lr.index.author_objects()[0].get_primary_title(
                    "en") if lr.index.author_objects() else "An Unknown Author",
                "en_text": lr.text('en').as_string()
            }
            commentaries += [Commentary(**c)]
        num_linked = (len(eng_linked_refs))

        return cls(
            ref=r.normal(),
            english_version=en_text,
            hebrew_version=he_text,
            num_linked=num_linked,
            linked_commentaries=commentaries
        )


class QuestionInterest(BaseModel):
    ref: str = Field(description="Identifier for the commentary text")
    question: str = Field(description="Questions that the source text seeks to answer")
    interest_rating: int = Field(description="Rating, from 1-10 for how interesting the question is")


class QuestionsInterest(BaseModel):
    # All the questions for a given commentary
    ref: str = Field(description="Identifier for the commentary text")
    questions_interest: List[QuestionInterest] = Field(description="Questions that the source text seeks to answer, and how interesting they are.")


class AllSourceQuestions(BaseModel):
    ref: str = Field(description="Identifier for the base text")
    all_questions: List[QuestionsInterest] = Field(description="Questions that the commentaries on the source text seek to answer, and how interesting they are.")


# Models used in find_top_questions, choose_sources
class QuestionGrouping(BaseModel):
    question: str = Field(description="Grouped question")
    included_commentaries: List[str] = Field(description="Ref indentifiers for the original questions that this one includes")


class QuestionGroupings(BaseModel):
    question_groupings: List[QuestionGrouping] = Field(description="Grouped questions")


class CommentSummary(BaseModel):
    commentaryRef: str = Field(description="Ref Identifier for the commentary text")
    summaryText: str = Field(description="Summary of the commentary")


class AnsweredQuestion(BaseModel):
    question: str = Field(description="Question being addressed")
    commentaries: List[CommentSummary] = Field(description="Commentaries that address the question")


class AnsweredQuestions(BaseModel):
    ref: str = Field(description="Identifier for the base text")
    questions: List[AnsweredQuestion] = Field(description="Questions that the commentaries seek to answer")


class GeminiApproach(BaseModel):
    approach: str = Field(description="Approach taken to the question by this text")
    ref: str = Field(description="Identifier for the related text")
    author: str = Field(description="Author of the related text")
    source: str = Field(description="text of the related text")


class GeminiQuestion(BaseModel):
    question: str = Field(description="Question being addressed")
    sources: List[GeminiApproach] = Field(description="The approaches and sources that address this question")


class GeminiQuestions(BaseModel):
    ref: str = Field(description="Identifier for the base text")
    questions: List[GeminiQuestion] = Field(description="Questions that the commentaries seek to answer")


class AnswerEvaluation(BaseModel):
    eval: int = Field(description="Evaluation of the summary accuracy, from 1-10, 1 being least accurate, 10 being most accurate")



def gemini_burrito(base: BaseText, readerProfile: str) -> AnsweredQuestions:
    parser = PydanticOutputParser(pydantic_object=GeminiQuestions)

    commentary_str = ""
    for commentary in base.linked_commentaries:
        commentary_str += f"{commentary.ref} ({commentary.author}) - {commentary.en_text}\n###\n"

    msg = f"""Sefaria is building an interface to help people explore the range of texts related to a base text. The user will read the base text, and then be offered 3 paths - we envision three questions. Each question will lead to a selection of about 3 short teasers. Each teaser introduces a primary text.

The base text is {base.ref}. Here is the text, in English and Hebrew:
'{base.english_version}'
'{base.hebrew_version}'

Please review the provided related texts, which are separated with ### below, and each of which begins with its identifying reference, author in parenthesis, then has a dash, then the text itself. 

First, find three questions that are addressed by the texts. We are looking for questions that would be interested to a {readerProfile}, where the related texts have divergent approaches or conclusions, preferably three or more divergent approaches or conclusions. 
Please show your reasoning, and then output the three questions as a numbered list. 

Then, please repeat each question and name the approaches taken to the question and list the identifying references for the sources that take that approach. 

Finally please choose one related text that exemplifies each approach.  
---
{commentary_str}
---
Return the data in the following format:
{parser.get_format_instructions()}
"""

    chain = gemini_llm | parser

    # print(msg)
    questions = chain.invoke(msg)

    parser = PydanticOutputParser(pydantic_object=AnsweredQuestions)
    msg = f"""Sefaria is building an interface to help people explore the range of texts related to a base text. The user will read the base text, and then be offered 3 paths - we envision three questions. Each question will lead to a selection of about 3 short teasers. Each teaser introduces a primary text.

The base text is {base.ref}. Here is the text, in English and Hebrew:
'{base.english_version}'
'{base.hebrew_version}'
    
Below are three questions that have been chosen to be addressed by the related texts.  For each question, the approaches taken to the question are listed, along with the sources that take that approach.  
For each source please write a teaser for that text that frames the source in the context of the base text, the question, and the approach taken.
The teaser should be a sentence or two, and should be written in a way that would be interesting to a {readerProfile}.
Please start the teaser with the name of the author of the related text and then a verb, like 'Rabbi Yonah believes'.  If the same commentator responds to the same question twice in a row, use something like 'He also' for the second one, instead of the name of the author.  If the author name is not known, use the name of the text in place of the author name. 
---
Here are the questions and approaches:
{questions.json()}
---
{parser.get_format_instructions()}
    """
    chain = gemini_llm | parser

    return chain.invoke(msg)


# Phase 1: Iterate over sources and get LLM response for each source
def analyze_sources(base: BaseText, readerProfile: str) -> AllSourceQuestions:
    parser = PydanticOutputParser(pydantic_object=QuestionsInterest)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
             "Given a base text and a commentary on that text, your job is to identify the questions about the base text that the commentary seeks to answer.\n"
             "Additionally, you are being asked to determine how interesting those questions and answers would be to a {readerProfile}." ),
        ("human", "In the context of this Mishnah from Pirkei Avot:\n\n" 
             "Original:\n" 
             "{base.hebrew_version}\n\n"
             "Translation:\n"
             "{base.english_version}\n\n"
             "This text is from the book {commentary.index_title} by {commentary.author}.\n"
             "The ref for this commentary is '{commentary.ref}'.\n"
             "Here is the commentary:\n\n"
             "{commentary.en_text}\n\n"
             "What questions does this seek to answer? And would the answer be compelling or interesting to a {readerProfile}?\n"
             "{format_instructions}")])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions(), readerProfile=readerProfile, base=base)
    chain = claude_opus_llm | parser

    all_qs = []
    for commentary in base.linked_commentaries:
        print(f"Text: {commentary.ref}")
        prompt_text = prompt.format(commentary=commentary)
        question_interests = chain.invoke(prompt_text)
        all_qs.append(question_interests)

    return AllSourceQuestions(ref=base.ref, all_questions=all_qs)


# Phase 2: Find the most discussed and interesting questions
def find_top_questions(base: BaseText, question_interests: AllSourceQuestions, readerProfile: str) -> QuestionGroupings:
    all_questions = ""
    for q in question_interests.all_questions:
        for p in q.questions_interest:
            all_questions += f"Ref: {q.ref} :: Question: {p.question} (Interest: {p.interest_rating}/10)\n"

    parser = PydanticOutputParser(pydantic_object=QuestionGroupings)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
            "One of your colleagues has reviewed a set of texts related to {base.ref}, identified the questions that each text addresses, and given a score of 1 to 10 on how interesting that question and answer would be to a {readerProfile}.\n"
            "Your job is to group together similar questions, and select 3 of the grouped questions that are different from each other, and together would offer interesting and various paths of learning for a {readerProfile}.\n"
            "Presume that the student has read the original text.  Please do not select questions that just repeat questions asked in the text."
            "The language of the questions differ, but they can be grouped into similar themes.\n"
            "Please phrase the resulting grouped questions in clear everyday language, in a concise form - 10 words or less.\n"       
            "Please return 3 groupings.  Include all of the unique indentifiers of commentaries that address this question.\n"
        ),
        ("human", "In the context of this Mishnah from Pirkei Avot:\n\n" 
             "Original:\n" 
             "{base.hebrew_version}\n\n"
             "Translation:\n"
             "{base.english_version}\n\n"
             "Here's the list of questions and their interest ratings:\n"
             "{all_questions}\n\n"     
             "{format_instructions}")])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions(), readerProfile=readerProfile, base=base, all_questions=all_questions)
    chain = claude_opus_llm | parser
    return chain.invoke(prompt.format())


# Phase 3: For each of the selected questions, choose three sources to highlight
def choose_sources(base: BaseText, question_groupings: QuestionGroupings, readerProfile: str) -> QuestionGroupings:
    new_question_groupings = []
    for question_grouping in question_groupings.question_groupings:
        all_sources = ""
        for r in question_grouping.included_commentaries:
            ref = Ref(r)
            all_sources += f"Ref: {ref.normal()}\nText: {ref.text('en').as_string()}\n\n"

        parser = PydanticOutputParser(pydantic_object=QuestionGrouping)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
                       "You are helping prepare materials for a class on {base.ref}.\n"
                       "You have been given a question addressed by texts related to {base.ref}.\n"
                       "Your job is to select three of the related texts that address this question.  As much as possible, the three texts should each take a different approach to addressing the question.\n"
                       "You are provided with the base text, the question being asked, and a list of the related texts.\n"
                       "The related texts chosen should be interesting and compelling to a {readerProfile}.\n"
                       "The commentaries should be diverse and offer different perspectives on the question.\n"
                       "Please return the original question and the ref identifiers for the three sources that best fulfill these criteria.\n"
                 ),
            ("human", "In the context of this Mishnah from Pirkei Avot:\n\n" 
                 "Original:\n" 
                 "{base.hebrew_version}\n\n"
                 "Translation:\n"
                 "{base.english_version}\n\n"
                 "The question being addressed is:"
                 "{question_grouping.question}\n\n"     
                 "Here's the list of sources:\n"
                 "{all_sources}\n\n"     
                 "{format_instructions}")])
        prompt = prompt.partial(format_instructions=parser.get_format_instructions(), readerProfile=readerProfile, base=base, question_grouping=question_grouping, all_sources=all_sources)
        chain = claude_opus_llm | parser
        new_question_groupings += [chain.invoke(prompt.format())]
    return QuestionGroupings(question_groupings=new_question_groupings)


# Phase 4: For each of the highlighted sources, summarize the source
def summarize_sources(base: BaseText, question_groupings: QuestionGroupings, readerProfile: str) -> AnsweredQuestions:
    answered_questions = []
    for question_grouping in question_groupings.question_groupings:
        all_sources = ""
        for r in question_grouping.included_commentaries:
            ref = Ref(r)
            all_sources += f"Ref: {ref.normal()}\nText: {ref.text('en').as_string()}\n\n"

        all_sources = ""
        for r in question_grouping.included_commentaries:
            ref = Ref(r)
            all_sources += f"Ref: {ref.normal()}\nAuthor: {ref.index.author_objects()[0].get_primary_title('en') if ref.index.author_objects() else 'An Unknown Author'}\nText: {ref.text('en').as_string()}\n\n"

        parser = PydanticOutputParser(pydantic_object=AnsweredQuestion)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
                       "You are helping prepare materials about {base.ref}.  You will be summarizing texts that address the question '{question_grouping.question}'.\n"
                       "You will be provided with three related texts that address this question.\n"
                       "Your job is to summarize how each of these texts addresses the question.  The summary should be phrased in a way that is interesting and compelling to a {readerProfile}.  Summaries should be in modern conversational language; they should not reflect the eccentricities of the source text.\n"
                       "For each text please write a short summary of fewer than 30 words that encapsulates the related text as it relates to the base text and the question.  In the event that two of the texts are similar, their summaries should accentuate what makes them different from each other.\n"
                       "The summary will start with the name of the author of the related text and then a verb, like 'Rabbi Yonah explains'.  If the same commentator repeats immediately following, begin with 'He also' instead of the name of the author.  If the author name is not known, use the name of the text in place of the author name.\n" 
                       "Please return the original question, the ref identifiers for the three sources, and the summary of each.\n"
                 ),
            ("human", "In the context of this Mishnah from Pirkei Avot:\n\n" 
                 "Original:\n" 
                 "{base.hebrew_version}\n\n"
                 "Translation:\n"
                 "{base.english_version}\n\n"
                 "The question being addressed is:"
                 "{question_grouping.question}\n\n"     
                 "Here's the list of sources:\n"
                 "{all_sources}\n\n"     
                 "{format_instructions}")])
        prompt = prompt.partial(format_instructions=parser.get_format_instructions(), readerProfile=readerProfile, base=base, question_grouping=question_grouping, all_sources=all_sources)
        chain = claude_opus_llm | parser
        answered_questions += [chain.invoke(prompt.format())]
    return AnsweredQuestions(questions=answered_questions, ref=base.ref)


def create_answer_data(base_text: BaseText, all_source_questions: AllSourceQuestions, profile: str):
    top_questions = find_top_questions(base_text, all_source_questions, profile)
    chosen_sources = choose_sources(base_text, top_questions, profile)
    return summarize_sources(base_text, chosen_sources, profile)



def pickle_data(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def pkl_file(ref: Ref, role: str) -> str:
    return f"{role}-{ref.url()}.pkl"


def load_or_create_data(ref, role, data_creation_func, *args):
    file_path = pkl_file(ref, role)
    if os.path.exists(file_path):
        return load_pickle(file_path)
    else:
        data = data_creation_func(*args)
        pickle_data(file_path, data)
        return data


def repair_pronouns(book_ref):
    # Currently, this is a one-off function to repair a specific issue with the data.
    # It logs, but doesn't change anything.

    for ref in book_ref.index.all_segment_refs():
        guide = Guide().load_by_ref(ref)
        if not guide:
            print("No " + ref.normal())
            continue
        for question in guide.questions:
            answers = question["commentaries"]
            previous_comment_ref = None
            for answer in answers:
                if answer["commentaryRef"].startswith("Pirkei Avot"):
                    print(ref.normal())
                    print("Self Reference! " + answer["commentaryRef"])
                if answer["summaryText"].startswith("He "):
                    current_comment_ref = Ref(answer["commentaryRef"])
                    if previous_comment_ref is None or current_comment_ref.index.title != previous_comment_ref.index.title:
                        print(ref.normal())
                        print(previous_comment_ref.normal() if previous_comment_ref else "None")
                        print(current_comment_ref.normal())
                        print()
                previous_comment_ref = Ref(answer["commentaryRef"])


def check_all_summaries():
    parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
                   "You have been given a base text, a question addressed, a commentary, and a summary of that commentary.\n"
                   "Your job is to determine if the summary accurately reflects the commentary.\n"
                   "Please rate the accuracy of the summary on a scale of 1 to 10, with 1 being the least accurate and 10 being the most accurate.\n"
                   "Please provide output just in the as a rating number, in the JSON format specified.\n"
             ),
        ("human", "In the context of this Mishnah from Pirkei Avot:\n\n"
                "Original:\n"
                "{base_text_he}\n"
                "Translation:\n"
                "{base_text_en}\n\n"
                "The question being addressed is:\n"
                "{question}\n\n"
                "Here is the commentary:\n"
                "{comment_text_en}\n\n"
                "Here is the summary:\n"
                "{summary}\n\n"  
                "{format_instructions}\n"
                "Please provide output just in the as a rating number, in the JSON format specified.\n"
         )])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | claude_opus_llm | parser

    for ref in Ref("Pirkei Avot").index.all_segment_refs():
        guide = Guide().load_by_ref(ref)
        if not guide:
            print(f"{ref.normal()},X,X,No Guide")
            continue
        for x, question_group in enumerate(guide.questions):
            q_number = x + 1
            answers = question_group["commentaries"]
            for y, answer in enumerate(answers):
                a_number = y + 1
                comment_ref = Ref(answer["commentaryRef"])

                sources = {
                    "base_text_en": ref.text('en').as_string(),
                    "comment_text_en": comment_ref.text('en').as_string(),
                    "base_text_he": ref.text('he').as_string(),
                    "comment_text_he": comment_ref.text('he').as_string(),
                    "question": question_group["question"],
                    "summary": answer["summaryText"],
                }
                try:
                    evaluation = chain.invoke(sources)
                    print(f"{ref.normal()},{q_number},{a_number},{evaluation.eval}")
                except Exception as e:
                    print(f"{ref.normal()},{q_number},{a_number},error,{e}")


def check_all_question_answer_pairs():
    # Do the answers relate to the questions asked?
    parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Jewish scholar knowledgeable in all Torah and Jewish texts, and an expert teacher of texts.\n"
                   "You have been given a base text, a question about that text, and an an answer to that question, based on a source.\n"
                   "Your job is to determine if the answer provided is in fact an answer to the question.\n"
                   "Please rate the relatedness of the answer on a scale of 1 to 10, with 1 being the least accurate and 10 being the most accurate.\n"
                   "Please provide output just in the as a rating number, in the JSON format specified.\n"
             ),
        ("human", "In the context of this Mishnah from Pirkei Avot:\n\n"
                "Original:\n"
                "{base_text_he}\n"
                "Translation:\n"
                "{base_text_en}\n\n"
                "The question being addressed is:\n"
                "{question}\n\n"
                "Here is the answer:\n"
                "{summary}\n\n---\n\n"  
                "Please provide output just in the as a rating number, in the JSON format specified.\n"
                "{format_instructions}\n"

         )])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | claude_opus_llm | parser

    for ref in Ref("Pirkei Avot").index.all_segment_refs():
        guide = Guide().load_by_ref(ref)
        if not guide:
            print(f"{ref.normal()},X,X,No Guide")
            continue
        for x, question_group in enumerate(guide.questions):
            q_number = x + 1
            answers = question_group["commentaries"]
            for y, answer in enumerate(answers):
                a_number = y + 1
                comment_ref = Ref(answer["commentaryRef"])

                sources = {
                    "base_text_en": ref.text('en').as_string(),
                    "comment_text_en": comment_ref.text('en').as_string(),
                    "base_text_he": ref.text('he').as_string(),
                    "comment_text_he": comment_ref.text('he').as_string(),
                    "question": question_group["question"],
                    "summary": answer["summaryText"],
                }
                try:
                    evaluation = chain.invoke(sources)
                    print(f"{ref.normal()},{q_number},{a_number},{evaluation.eval}")
                except Exception as e:
                    print(f"{ref.normal()},{q_number},{a_number},error,{e}")


def generate_guide(book_ref, profile):
    # Run one iteration of a loop every minute  (60 seconds)
    for ref in book_ref.index.all_segment_refs():
        start_time = datetime.datetime.now()

        try:
            file_path = pkl_file(ref, "gemini_answers")
            if os.path.exists(file_path):
                continue
            else:
                print(f"Processing: {ref.normal()} @{start_time}")
                base_text = BaseText.from_ref(ref, [Ref("Jastrow")])
                answered_questions = gemini_burrito(base_text, profile)
                pickle_data(file_path, answered_questions)
                Guide().load_from_dict(answered_questions.dict()).save()
        except Exception as e:
            print(f"Error generating {ref.normal()}: {e}")

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        time.sleep(60 - elapsed_time if elapsed_time < 60 else 0)

def leftovers():
    profile = "modern person who is not particularly educated in Jewish texts"
    for tref in ["Pirkei Avot 5:8", "Pirkei Avot 6:1"]:
        ref = Ref(tref)
        base_text = BaseText.from_ref(ref, [Ref("Jastrow")])
        answered_questions = gemini_burrito(base_text, profile)

def main():
    profile = "modern person who is not particularly educated in Jewish texts"
    book_ref = Ref("Pirkei Avot")

    generate_guide(book_ref, profile)
    # repair_pronouns(book_ref)
    # check_all_summaries()
    # check_all_question_answer_pairs()


if __name__ == "__main__":
    main()

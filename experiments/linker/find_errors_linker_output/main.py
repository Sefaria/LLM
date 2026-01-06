"""
Citation Classifier and Book Title Extractor for Sefaria

This script uses LangChain to:
1. Classify if a citation/reference is complete (has all information to locate the exact source)
   - Quotations with missing punctuation are still considered complete if they have identifying information
2. Extract the top 3 Sefaria book titles from complete citations
3. Analyze references to extract:
   - Completeness status
   - The title as it appears in the reference
   - The canonical Hebrew title from Sefaria

Supports both Hebrew and English citations and references.
"""

import sys
from pathlib import Path

# Add app directory to path for imports
app_path = Path(__file__).parent.parent.parent.parent / "app"
sys.path.insert(0, str(app_path))

from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
from util.general import run_parallel
from typing import Any, TypeVar
from pydantic import BaseModel, Field
import pymongo
import json

T = TypeVar('T', bound=BaseModel)



class CitationCompleteness(BaseModel):
    """Model for citation completeness classification"""
    is_complete: bool = Field(description="Whether the citation is complete and has all information to locate the exact source")
    reasoning: str = Field(description="Brief explanation of why the citation is or isn't complete")


class BookTitlePrediction(BaseModel):
    """Model for book title predictions"""
    book_titles: list[str] = Field(description="Top 3 predicted Sefaria book titles", max_length=3)
    confidence: str = Field(description="Overall confidence level: high, medium, or low")


class ReferenceAnalysis(BaseModel):
    """Model for complete reference analysis including canonical Hebrew title"""
    reasoning: str = Field(description="Explanation of the analysis")
    is_complete: bool = Field(description="Whether the reference is complete (quotations with missing punctuation are still complete)")
    quoted_title: str = Field(description="The title as it appears in the reference, or 'N/A' if no title found")
    canonical_title: str = Field(description="The canonical Hebrew title from Sefaria, or 'N/A' if not found")


class CitationAnalyzer:
    """Analyzes citations to determine completeness and extract book titles"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the citation analyzer
        
        Args:
            model_name: The LLM model to use (e.g., 'gpt-4o', 'gpt-4', 'claude-3-5-sonnet-20241022')
            temperature: Temperature for LLM responses (0.0 for most deterministic)
        """
        if "claude" in model_name:
            self.llm = ChatAnthropic(model=model_name, temperature=temperature)
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    def _parse_json_response(self, response_content: str, model_class: type[T]) -> T:
        """
        Parse JSON response from LLM, handling markdown code blocks
        
        Args:
            response_content: Raw response content from LLM
            model_class: Pydantic model class to parse into
            
        Returns:
            Parsed Pydantic model instance
        """
        content = response_content.strip()
        
        # Remove markdown code block markers
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Use Pydantic's model_validate_json for parsing
        try:
            return model_class.model_validate_json(content)
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Content: {content}")
            raise
    
    def classify_citation(self, citation: str) -> CitationCompleteness:
        """
        Classify whether a citation is complete
        
        Args:
            citation: The citation text in Hebrew or English
            
        Returns:
            CitationCompleteness object with is_complete and reasoning
        """
        system_prompt = """You are an expert in Jewish texts and citations. Your task is to determine if a citation is COMPLETE.

A COMPLETE citation contains ALL the information needed to locate the EXACT location in a specific book WITHOUT relying on external context (like previous citations or surrounding text).

Examples of COMPLETE citations:
- "Genesis 1:1" (book + chapter + verse)
- "בראשית א:א" (book + chapter + verse in Hebrew)
- "Talmud Bavli Berakhot 2a" (book + tractate + page)
- "Rashi on Genesis 1:1" (commentary + book + chapter + verse)
- "Mishnah Berakhot 1:1" (collection + tractate + chapter + mishnah)

Examples of INCOMPLETE citations:
- "1:1" (missing book name - could be any book)
- "verse 5" (missing book and chapter)
- "there" (pronoun, needs context)
- "ibid." or "שם" (refers to previous citation)
- "the next chapter" (relative reference)
- "above" or "below" (relative reference)

Respond in JSON format with:
- "is_complete": true/false
- "reasoning": brief explanation

Be strict: if ANY information is missing to pinpoint the exact location, mark it as incomplete."""

        user_prompt = f"""Analyze this citation: "{citation}"

Is this citation complete? Does it have all the information to find the exact location in a book without external context?"""

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt)
        ]
        
        response = self.llm(messages)
        
        try:
            return self._parse_json_response(response.content, CitationCompleteness)
        except Exception:
            # Fallback parsing if not perfect JSON
            content_lower = response.content.lower()
            is_complete = "true" in content_lower or '"is_complete": true' in content_lower
            return CitationCompleteness(
                is_complete=is_complete,
                reasoning=response.content
            )
    
    def get_book_titles(self, citation: str) -> BookTitlePrediction:
        """
        Extract top 3 Sefaria book titles from a complete citation
        
        Args:
            citation: The complete citation text in Hebrew or English
            
        Returns:
            BookTitlePrediction object with book_titles and confidence
        """
        system_prompt = """You are an expert in Jewish texts and the Sefaria library. Your task is to identify the top 3 most likely Sefaria book titles that match a given citation.

IMPORTANT: Use EXACT Sefaria spelling for book titles. Here are common examples:

Torah/Tanakh:
- "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"
- "Joshua", "Judges", "I Samuel", "II Samuel", "I Kings", "II Kings"
- "Isaiah", "Jeremiah", "Ezekiel"
- "Psalms", "Proverbs", "Job"

Talmud Bavli:
- "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah"
- Format: Just the tractate name, NOT "Talmud Bavli Berakhot"

Mishnah:
- "Mishnah Berakhot", "Mishnah Peah", "Mishnah Shabbat"

Rashi:
- "Rashi on Genesis", "Rashi on Berakhot"

Midrash:
- "Bereshit Rabbah", "Shemot Rabbah", "Vayikra Rabbah"
- "Midrash Tanchuma"

Other:
- "Shulchan Arukh, Orach Chayim"
- "Mishneh Torah, Hilchot Shabbat"

For Hebrew citations, translate to the English Sefaria title.

Respond in JSON format with:
- "book_titles": array of exactly 3 book title predictions (most likely first)
- "confidence": "high", "medium", or "low"

Order by likelihood. Use exact Sefaria spelling."""

        user_prompt = f"""What are the top 3 most likely Sefaria book titles for this citation?

Citation: "{citation}"

Provide exactly 3 predictions using Sefaria's exact spelling."""

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt)
        ]
        
        response = self.llm(messages)
        
        try:
            return self._parse_json_response(response.content, BookTitlePrediction)
        except Exception as e:
            # Fallback parsing - try to extract titles
            print(f"Failed to parse JSON for book titles: {e}")
            return BookTitlePrediction(
                book_titles=["Unable to parse", "response", "properly"],
                confidence="low"
            )
    
    def analyze_reference_with_canonical_title(self, reference: str) -> ReferenceAnalysis:
        """
        Analyze a reference to check completeness, extract title, and get canonical Hebrew title

        Args:
            reference: The reference text in Hebrew or English

        Returns:
            ReferenceAnalysis object with reasoning, is_complete, quoted_title, and canonical_title
        """
        system_prompt = """You are an expert in Jewish texts and citations. Your task is to analyze a reference and provide:

1. **Completeness check**: Determine if the reference is COMPLETE (has all information to locate the exact source).
   - IMPORTANT: A quotation with missing punctuation is STILL considered complete if it has enough information to identify the source.
   - Examples of COMPLETE references:
     * "Genesis 1:1" (book + chapter + verse)
     * "בראשית א:א" (book + chapter + verse in Hebrew)
     * "Talmud Bavli Berakhot 2a"
     * "Rashi on Genesis 1:1"
     * A direct quotation that clearly identifies a specific source (even if punctuation is missing)
   - Examples of INCOMPLETE references:
     * "1:1" (missing book name)
     * "verse 5" (missing book and chapter)
     * "ibid." or "שם" (refers to previous citation)
     * "there" or "above" (relative reference)

2. **Title extraction**: If the book/text title it appears in the reference:
   - Extract exactly how the title appears in the text
   - Return "N/A" if no title is found

3. **Canonical Hebrew title**: If title was found, provide the Sefaria canonical Hebrew title for the book.
   - Use your knowledge of Sefaria's Hebrew titles
   - Examples:
     * "Genesis" → "בראשית"
     * "Berakhot" (Talmud) → "ברכות"
     * "Rashi on Genesis" → "רש\"י על בראשית"
     * "Mishnah Berakhot" → "משנה ברכות"
   - Return "N/A" if the title cannot be determined or matched

Respond in JSON format with:
- "reasoning": Brief explanation of your analysis
- "is_complete": true/false (remember: quotations with missing punctuation can still be complete)
- "quoted_title": The title as it appears in the reference, or "N/A"
- "canonical_title": The canonical Hebrew title from Sefaria, or "N/A"
"""

        user_prompt = f"""Analyze this reference: "{reference}"

Provide your analysis following the format specified."""

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt)
        ]

        response = self.llm(messages)

        try:
            result = self._parse_json_response(response.content, ReferenceAnalysis)
            return result
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Content: {response.content}")
            # Return a fallback response
            return ReferenceAnalysis(
                reasoning=f"Failed to parse response: {str(e)}",
                is_complete=False,
                quoted_title="N/A",
                canonical_title="N/A"
            )

    def analyze_citation(self, citation: str) -> dict[str, Any]:
        """
        Complete analysis: classify completeness and extract book titles if complete

        Args:
            citation: The citation text in Hebrew or English

        Returns:
            Dictionary with completeness and book titles (if applicable)
        """
        # Step 1: Classify if citation is complete
        completeness = self.classify_citation(citation)

        result: dict[str, Any] = {
            "citation": citation,
            "is_complete": completeness.is_complete,
            "reasoning": completeness.reasoning,
            "book_titles": None
        }

        # Step 2: If complete, extract book titles
        if completeness.is_complete:
            book_prediction = self.get_book_titles(citation)
            result["book_titles"] = {
                "titles": book_prediction.book_titles,
                "confidence": book_prediction.confidence
            }

        return result


def test():
    """Example usage of the CitationAnalyzer"""

    # Initialize analyzer
    analyzer = CitationAnalyzer(model_name="claude-sonnet-4-5-20250929", temperature=0.0)

    # Test references for the new analyze_reference_with_canonical_title method
    test_references = [
        "Genesis 1:1",
        "בראשית א:א",
        "Rashi on Exodus 12:2",
        "1:1",  # Incomplete - no book
        "Berakhot 2a",
        "verse 5",  # Incomplete
        "Mishnah Berakhot 1:1",
        "שם",  # Incomplete - means "ibid."
        "Shulchan Arukh, Orach Chayim 1:1",
        "the next chapter",  # Incomplete
        '"In the beginning God created" (Genesis 1:1)',  # Quotation with reference
        'ויהי ערב ויהי בקר',  # Quotation without explicit reference - incomplete
    ]

    print("=" * 80)
    print("REFERENCE ANALYSIS WITH CANONICAL HEBREW TITLES")
    print("=" * 80)

    for reference in test_references:
        print(f"\n{'=' * 80}")
        print(f"Reference: {reference}")
        print("-" * 80)

        result = analyzer.analyze_reference_with_canonical_title(reference)

        print(f"Complete: {result.is_complete}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Quoted Title: {result.quoted_title}")
        print(f"Canonical Hebrew Title: {result.canonical_title}")

        print("=" * 80)

    # Also demonstrate JSON output
    print("\n" + "=" * 80)
    print("EXAMPLE JSON OUTPUT")
    print("=" * 80)
    example_ref = "Genesis 1:1"
    result = analyzer.analyze_reference_with_canonical_title(example_ref)
    print(json.dumps({
        "reasoning": result.reasoning,
        "is_complete": result.is_complete,
        "quoted_title": result.quoted_title,
        "canonical_title": result.canonical_title
    }, ensure_ascii=False, indent=2))
    print("=" * 80)


def main():
    with open('texts.json') as fp:
        texts = json.load(fp)
    analyzer = CitationAnalyzer(model_name="claude-sonnet-4-5-20250929", temperature=0.0)
    maximum = 766746
    refs = texts[:maximum]
    results = []
    for r, ref in enumerate(refs):
        try:
            result = analyzer.analyze_reference_with_canonical_title(ref)
        except Exception as e:
            print('Exception', r, e)
        results.append({
            'ref': ref,
            "is_complete": result.is_complete,
            "reasoning": result.reasoning,
            "quoted_title": result.quoted_title,
            "canonical_title": result.canonical_title
        })
    # results = run_parallel(refs, analyzer.analyze_reference_with_canonical_title, 50)
    # results = [{
    #             'ref': ref,
    #             "is_complete": analysis.is_complete,
    #             "reasoning": analysis.reasoning,
    #             "quoted_title": analysis.quoted_title,
    #             "canonical_title": analysis.canonical_title
    #         } for analysis, ref in zip(results, refs)]

    with open('results.json', 'w', encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)
    print(f"Total analyzed: {len(results)}")
    print(f"Complete references: {len([r for r in results if r['is_complete']])}")
    print(f"With canonical titles: {len([r for r in results if r['canonical_title'] != 'N/A'])}")


def make_spans_file():
    client = pymongo.MongoClient('localhost', 27017, username='', password='')
    db = client['sefaria']
    col = db.linker_output
    pipeline = [
        {"$unwind": "$spans"},
        {"$match": {
            "spans.type": "citation",
            "spans.failed": True,
        }},
        {"$replaceRoot": {"newRoot": {"text": "$spans.text"}}}
    ]
    texts = list(set([doc["text"] for doc in col.aggregate(pipeline)]))
    with open('texts.json', 'w', encoding='utf-8') as fp:
        json.dump(texts, fp, ensure_ascii=False)


if __name__ == "__main__":
    main()


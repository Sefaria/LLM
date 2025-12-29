"""
Citation Classifier and Book Title Extractor for Sefaria

This script uses LangChain to:
1. Classify if a citation is complete (has all information to locate the exact source)
2. Extract the top 3 Sefaria book titles from complete citations

Supports both Hebrew and English citations.
"""

import sys
from pathlib import Path

# Add app directory to path for imports
app_path = Path(__file__).parent.parent.parent.parent / "app"
sys.path.insert(0, str(app_path))

from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
from typing import Any
from pydantic import BaseModel, Field


class CitationCompleteness(BaseModel):
    """Model for citation completeness classification"""
    is_complete: bool = Field(description="Whether the citation is complete and has all information to locate the exact source")
    reasoning: str = Field(description="Brief explanation of why the citation is or isn't complete")


class BookTitlePrediction(BaseModel):
    """Model for book title predictions"""
    book_titles: list[str] = Field(description="Top 3 predicted Sefaria book titles", max_length=3)
    confidence: str = Field(description="Overall confidence level: high, medium, or low")


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
    
    def _parse_json_response(self, response_content: str, model_class: type[BaseModel]) -> BaseModel:
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


def main():
    """Example usage of the CitationAnalyzer"""
    
    # Initialize analyzer
    analyzer = CitationAnalyzer(model_name="gpt-4o", temperature=0.0)
    
    # Test citations (mix of complete and incomplete, Hebrew and English)
    test_citations = [
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
    ]
    
    print("=" * 80)
    print("CITATION ANALYSIS RESULTS")
    print("=" * 80)
    
    for citation in test_citations:
        print(f"\n{'=' * 80}")
        print(f"Citation: {citation}")
        print("-" * 80)
        
        result = analyzer.analyze_citation(citation)
        
        print(f"Complete: {result['is_complete']}")
        print(f"Reasoning: {result['reasoning']}")
        
        if result['book_titles']:
            print(f"\nTop 3 Book Titles:")
            for i, title in enumerate(result['book_titles']['titles'], 1):
                print(f"  {i}. {title}")
            print(f"Confidence: {result['book_titles']['confidence']}")
        
        print("=" * 80)


if __name__ == "__main__":
    main()


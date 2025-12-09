# Citation Classifier and Book Title Extractor

This LangChain script analyzes Sefaria citations to determine if they are complete and extracts the most likely book titles.

## Features

1. **Citation Completeness Classification**: Determines if a citation contains all information needed to locate the exact source without external context
2. **Book Title Extraction**: For complete citations, predicts the top 3 most likely Sefaria book titles
3. **Multilingual Support**: Works with both Hebrew and English citations

## Usage

### Basic Usage

```python
from main import CitationAnalyzer

# Initialize the analyzer
analyzer = CitationAnalyzer(model_name="gpt-4o", temperature=0.0)

# Analyze a single citation
result = analyzer.analyze_citation("Genesis 1:1")

print(f"Complete: {result['is_complete']}")
print(f"Reasoning: {result['reasoning']}")

if result['book_titles']:
    print("Top 3 Book Titles:")
    for i, title in enumerate(result['book_titles']['titles'], 1):
        print(f"  {i}. {title}")
    print(f"Confidence: {result['book_titles']['confidence']}")
```

### Running the Example Script

```bash
cd /Users/nss/sefaria/llm
python experiments/linker/find_errors_linker_output/main.py
```

## API Reference

### CitationAnalyzer

Main class for analyzing citations.

#### `__init__(model_name: str = "gpt-4o", temperature: float = 0.0)`

Initialize the analyzer.

**Parameters:**
- `model_name` (str): The LLM model to use. Options include:
  - `"gpt-4o"` (default)
  - `"gpt-4"`
  - `"claude-3-5-sonnet-20241022"`
- `temperature` (float): Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)

#### `classify_citation(citation: str) -> CitationCompleteness`

Classify whether a citation is complete.

**Parameters:**
- `citation` (str): The citation text in Hebrew or English

**Returns:**
- `CitationCompleteness` object with:
  - `is_complete` (bool): Whether the citation is complete
  - `reasoning` (str): Explanation of the classification

#### `get_book_titles(citation: str) -> BookTitlePrediction`

Extract top 3 Sefaria book titles from a complete citation.

**Parameters:**
- `citation` (str): The complete citation text

**Returns:**
- `BookTitlePrediction` object with:
  - `book_titles` (list[str]): Top 3 predicted book titles
  - `confidence` (str): "high", "medium", or "low"

#### `analyze_citation(citation: str) -> dict`

Complete analysis pipeline: classify completeness and extract book titles if complete.

**Parameters:**
- `citation` (str): The citation text

**Returns:**
- Dictionary with:
  - `citation` (str): Original citation
  - `is_complete` (bool): Whether the citation is complete
  - `reasoning` (str): Explanation
  - `book_titles` (dict | None): Book title predictions (if complete)

## Examples

### Complete Citations

```python
# English Torah citation
result = analyzer.analyze_citation("Genesis 1:1")
# is_complete: True
# book_titles: ["Genesis", "Rashi on Genesis", "Bereshit Rabbah"]

# Hebrew Torah citation
result = analyzer.analyze_citation("בראשית א:א")
# is_complete: True
# book_titles: ["Genesis", "Rashi on Genesis", "Bereshit Rabbah"]

# Commentary citation
result = analyzer.analyze_citation("Rashi on Exodus 12:2")
# is_complete: True
# book_titles: ["Rashi on Exodus", "Exodus", "Rashi on Genesis"]

# Talmud citation
result = analyzer.analyze_citation("Berakhot 2a")
# is_complete: True
# book_titles: ["Berakhot", "Rashi on Berakhot", "Mishnah Berakhot"]

# Mishnah citation
result = analyzer.analyze_citation("Mishnah Berakhot 1:1")
# is_complete: True
# book_titles: ["Mishnah Berakhot", "Berakhot", "Rashi on Berakhot"]
```

### Incomplete Citations

```python
# Missing book name
result = analyzer.analyze_citation("1:1")
# is_complete: False
# reasoning: "The citation '1:1' is incomplete because it lacks the book name..."

# Relative reference
result = analyzer.analyze_citation("verse 5")
# is_complete: False

# Hebrew "ibid."
result = analyzer.analyze_citation("שם")
# is_complete: False

# Relative reference
result = analyzer.analyze_citation("the next chapter")
# is_complete: False
```

## How It Works

### Step 1: Citation Completeness Classification

The script uses an LLM to determine if a citation is **complete**. A complete citation must have ALL information needed to locate the exact source WITHOUT external context.

**Complete citations include:**
- Book name + chapter + verse (e.g., "Genesis 1:1")
- Tractate + page (e.g., "Berakhot 2a")
- Commentary + book + location (e.g., "Rashi on Genesis 1:1")
- Collection + tractate + location (e.g., "Mishnah Berakhot 1:1")

**Incomplete citations include:**
- Missing book name (e.g., "1:1")
- Pronouns or relative references (e.g., "there", "ibid.", "שם")
- Contextual references (e.g., "the next chapter", "above")

### Step 2: Book Title Extraction

If the citation is complete, the script extracts the top 3 most likely Sefaria book titles using exact Sefaria spelling.

The LLM is trained on common Sefaria book title patterns:
- Torah/Tanakh: "Genesis", "Exodus", "Isaiah", "Psalms"
- Talmud Bavli: "Berakhot", "Shabbat" (just tractate name)
- Mishnah: "Mishnah Berakhot", "Mishnah Shabbat"
- Commentaries: "Rashi on Genesis", "Rashi on Berakhot"
- Midrash: "Bereshit Rabbah", "Midrash Tanchuma"
- Legal codes: "Shulchan Arukh, Orach Chayim", "Mishneh Torah, Hilchot Shabbat"

## Dependencies

- `langchain` - LLM framework
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `pydantic` - Data validation

## Environment Variables

Set your API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Notes

- The script uses temperature=0.0 by default for deterministic results
- JSON responses are automatically parsed, handling markdown code blocks
- Hebrew citations are translated to English Sefaria titles
- The script uses the `basic_langchain` custom wrapper for LLM calls


# SheetScorer - Jewish Study Sheet Analysis Tool

**SheetScorer** is a Python tool that uses **LLMs** to automatically analyze 
and score Jewish study sheets for reference relevance and title interest. 
It processes sheets, evaluates how well each cited reference
is discussed, and assigns engagement scores to sheet titles.

## Scores Extracted

- **Reference Discussion Scoring**: Analyzes how thoroughly each reference is discussed (**0-4 scale**)
- **Title Interest Scoring**: Evaluates how engaging sheet titles are to potential readers (**0-4 scale**)  
- **Creativity Assessment**: Computes creativity scores based on percentage of **user-generated content**. 
- **Title Interest Reason**: Explanation of title scoring. 
- **Language**: Language of the sheet [all the languages are supported not only he and en]. 

## Quick Start

```python
from sheet_scoring.sheet_scoring import score_one_sheet
from sefaria_llm_interface.sheet_scoring import SheetScoringInput

input_data = SheetScoringInput(
    sheet_id="123",
    title="Understanding Genesis Creation",
    expanded_refs=["Genesis 1:1", "Genesis 1:2"],
    sources=[
        {"outsideText": "This commentary explores..."},
        {"ref": "Genesis 1:1", "text": {"en": "In the beginning..."}, "comment": "Analysis here..."}
    ]
)

result = score_one_sheet(input_data)
print(f"Title score: {result.title_interest_level}")
print(f"Ref scores: {result.ref_scores}")
print(result)
```

## Scoring System

### Architecture

#### sheet_scoring (package)
- sheet_scoring.py - Main API with score_one_sheet() function 
- tasks.py - Celery task wrapper for async processing 
- text_utils.py - Content parsing and token counting utilities 
- openai_sheets_scorer.py - Core LLM scoring engine
- README.md

### Reference Discussion Levels

The tool evaluates how well each reference is discussed using a **0-4 scale**:

| Level | Description |
|-------|-------------|
| **0 - Not Discussed** | Reference is **quoted only**, no discussion or commentary |
| **1 - Minimal** | Mentioned only through **neighboring verses**, minimal engagement |
| **2 - Moderate** | Some discussion present with **basic commentary** |
| **3 - Significant** | **Substantial discussion** with detailed commentary |
| **4 - Central** | Reference is a **central focus** of the entire sheet |

### Title Interest Levels

Sheet titles are scored for **user engagement** on a **0-4 scale**:

| Level | Description |
|-------|-------------|
| **0 - Not Interesting** | **Off-topic** or unengaging for target users |
| **1 - Slight Relevance** | **Low appeal**, users unlikely to engage |
| **2 - Somewhat Interesting** | Users might **skim**, moderate appeal |
| **3 - Interesting** | Users **likely to open** and read |
| **4 - Very Compelling** | **Must-read content**, high engagement expected |

### Creativity Score

user_tokens / total_tokens - Higher = more original content vs canonical quotes.

### Language
ISO-639-1 language code of the sheet, and in case the sheet has no user generated content, the language code of the title.

## Data Structures
#### Input (SheetScoringInput)

```python
{
    "sheet_id": "123",
    "title": "Sheet title",
    "expanded_refs": ["Genesis 1:1", "Exodus 2:3"],
    "sources": [
        {"outsideText": "User commentary"},
        {"outsideBiText": {"en": "English", "he": "Hebrew"}},
        {"ref": "Genesis 1:1", "text": {"en": "Quote"}, "comment": "Analysis"}
    ]
}
```
#### Output (SheetScoringOutput)
```python
{
    "sheet_id": "123",
    "ref_levels": {"Genesis 1:1": 3, "Exodus 2:3": 2},      # Raw 0-4 scores
    "ref_scores": {"Genesis 1:1": 60.0, "Exodus 2:3": 40.0}, # Normalized %
    "title_interest_level": 3,
    "title_interest_reason": "Compelling theological question",
    "language": "en",
    "creativity_score": 0.75,
    "processed_datetime": "2025-01-31T10:30:00Z",
    "request_status": 1,  # 1=success, 0=failure
    "request_status_message": ""
}
```

## Configuration Options

### Initialization Parameters

```python
scorer = SheetScorer(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",                    # Default model
    max_prompt_tokens=128000,               # Input token budget
    token_margin=16384,                     # Reserved for output
    max_ref_to_process=800,                 # Max num of refs that can be processed 
    chunk_size=80                           # Refs per LLM call
)
```

The constants DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_INPUT_OUTPUT_TOKENS are model specific 
and can be found on the internet.

## Content Processing Strategy

The tool uses an **adjustable approach** for canonical quotations:

1. **Always includes** all user commentary and **original content**
2. **Conditionally includes** canonical quotes only if the **entire bundle** fits within token limits
and **add_full_commentary is set to True** 
3. **Truncates intelligently** using **LLM summarization** when content exceeds limits 

    1. ***LLM Summarization***: Uses secondary LLM to compress content while preserving key information 
    2. ***Reference Preservation***: Maintains all biblical reference tags during compression 
    3. ***Character Fallback***: Falls back to character-based truncation if summarization fails
## Grading Strategy 
Processed content is sent to LLM, together with references for grading: 

### Resilient Grading List Processing

- **Chunking**: Large reference lists are processed in **chunks** to stay within model limits
- **Overlap Handling**: Smart overlap between chunks prevents **reference boundary issues**

### Resilient Reference Grading

- **Primary attempt**: Process **all references together**
- **Fallback**: Split reference list in **half** and process **recursively**
- **Final fallback**: Assign **default score of 0** to problematic references


### Resilient score extraction 

Uses **OpenAI's function calling** feature with **strict schemas**:

#### Middle Chunk Scoring Schema 
```python
{
    "name": "score_references",
    "parameters": {
        "ref_levels": {
            "Genesis 1:1": {"type": "integer", "minimum": 0, "maximum": 4},
            # ... for each reference
        }
    }
}
```

#### Title Scoring Schema
```python
{
    "name": "score_title", 
    "parameters": {
        "language": {"type": "string"},
        "title_interest_level": {"type": "integer", "minimum": 0, "maximum": 4},
        "title_interest_reason": {"type": "string", "maxLength": 100}
    }
}
```


## Database Integration

Designed for **MongoDB integration** with expected document structure:

```python
{
    "id": "unique id",
    "title": "Sheet Title",
    "expandedRefs": ["Genesis 1:1", "Exodus 2:3"],
    # Additional sheet content fields...
}
```

## Output Fields

| Field                        | Description                                    |
|------------------------------|------------------------------------------------|
| **`ref_levels`**             | Raw **0-4 scores** for each reference          |
| **`ref_scores`**             | **Normalized percentage scores** (sum to 100%) |
| **`title_interest_level`**   | Title **engagement score** (0-4)               |
| **`title_interest_reason`**  | **Brief explanation** of title score           |
| **`language`**               | **Detected language code**                     |
| **`creativity_score`**       | **Percentage** of user-generated content       |
| **`processed_datetime`**     | **Processing timestamp**                       |
| **`request_status`**         | **Whether scoring succeded/failed**            |
| **`request_status_message`** | **The reason why scoring failed**              |




## Logging

**Comprehensive logging** for monitoring and debugging:

- **Info**: Processing decisions and **content statistics**
- **Warning**: **Score validation** and fallback usage  
- **Error**: **LLM failures** and processing errors

Configure logging level as needed:
```python
import logging
logging.getLogger('sheet_scorer').setLevel(logging.INFO)
```



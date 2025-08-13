# SheetScorer - Jewish Study Sheet Analysis Tool

**SheetScorer** is a Python tool that uses **LLMs** to automatically analyze and score Jewish study sheets for reference relevance and title interest. It processes sheets from **MongoDB**, evaluates how well biblical references are discussed, and assigns engagement scores to sheet titles.

## Scores Extracted

- **Reference Discussion Scoring**: Analyzes how thoroughly each biblical reference is discussed (**0-4 scale**)
- **Title Interest Scoring**: Evaluates how engaging sheet titles are to potential readers (**0-4 scale**)  
- **Creativity Assessment**: Computes creativity scores based on percentage of **user-generated content**. 
- **Title Interest Reason**: Explanation of title scoring. 

## Quick Start

```python
from sheet_scorer import SheetScorer

# Initialize scorer
scorer = SheetScorer(
    api_key="your-openai-api-key",
    model="gpt-4o-mini"
)

# Process a sheet
sheet_data = {
    "_id": "sheet123",
    "title": "Understanding Genesis Creation",
    "expandedRefs": ["Genesis 1:1", "Genesis 1:2", "Genesis 1:3"],
    # ... other sheet content
}

result = scorer.process_sheet_by_content(sheet_data)
print(result)
```

## Scoring System

### Reference Discussion Levels

The tool evaluates how well each biblical reference is discussed using a **0-4 scale**:

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

Calculated as the **percentage of user-generated content** versus all text (including quoted canonical text). Higher scores indicate more **original commentary** and analysis.

## Configuration Options

### Initialization Parameters

```python
scorer = SheetScorer(
    api_key="your-api-key",           # OpenAI API key
    model="gpt-4o-mini",              # Model to use
    max_prompt_tokens=128000,         # Maximum input tokens
    token_margin=16384,               # Reserved tokens for output
    max_ref_to_process=800,           # Maximum references to process
    chunk_size=80                     # References per chunk
)
```

### Key Constants

- **DEFAULT_MAX_OUTPUT_TOKENS**: **16384**
- **DEFAULT_CHUNK_SIZE**: **80** references per processing chunk
- **DEFAULT_MAX_INPUT_OUTPUT_TOKENS**: **128000** total token limit
- **MAX_CHUNK_OVERLAP**: **10** references overlap between chunks

## Core Methods

### **process_sheet_by_content(sheet, add_full_comment)**

**Main method** to process a complete sheet and return scores.

**Parameters:**
- `sheet` (**Dict**): **MongoDB** sheet document containing title, references, and content
- `add_full_comment` (**bool**): parameter that allows to add quotations text to input that LLM receives

**Returns:**
- **Dictionary** with scoring results or **None** if processing fails

**Example Output:**
```python
{
    "_id": "sheet123",
    "ref_levels": {"Genesis 1:1": 3, "Genesis 1:2": 2},
    "ref_scores": {"Genesis 1:1": 60.0, "Genesis 1:2": 40.0},
    "title_interest_level": 3,
    "title_interest_reason": "Compelling theological question",
    "language": "en",
    "creativity_score": 0.75,
    "processed_at": "2025-01-31T10:30:00Z"
}
```
! ref_scores is normalized version of ref_levels

### **get_gpt_scores(content, ref_names, title)**

**Core scoring method** that processes content and returns analysis.

**Parameters:**
- `content` (**str**): Sheet text content to analyze
- `ref_names` (**List[str]**): List of biblical references to score
- `title` (**str**): Sheet title to evaluate

## Content Processing Strategy

The tool uses an **adjustable approach** for canonical quotations:

1. **Always includes** all user commentary and **original content**
2. **Conditionally includes** canonical quotes only if the **entire bundle** fits within token limits
and **add_full_comment is set to True** 
3. **Truncates intelligently** using **LLM summarization** when content exceeds limits 
   4. ***LLM Summarization***: Uses secondary LLM to compress content while preserving key information 
   5. ***Reference Preservation***: Maintains all biblical reference tags during compression 
   6. ***Character Fallback***: Falls back to character-based truncation if summarization fails

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
    "_id": "unique_sheet_id",
    "title": "Sheet Title",
    "expandedRefs": ["Genesis 1:1", "Exodus 2:3"],
    # Additional sheet content fields...
}
```

## Output Fields

| Field                       | Description                                    |
|-----------------------------|------------------------------------------------|
| **`ref_levels`**            | Raw **0-4 scores** for each reference          |
| **`ref_scores`**            | **Normalized percentage scores** (sum to 100%) |
| **`title_interest_level`**  | Title **engagement score** (0-4)               |
| **`title_interest_reason`** | **Brief explanation** of title score           |
| **`language`**              | **Detected language code**                     |
| **`creativity_score`**      | **Percentage** of user-generated content       |
| **`processed_datetime`**    | **Processing timestamp**                       |
| **`request_status`**        | **Whether scoring succeded/failed**            |
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


## Language Support

Supports **automatic detection** and processing of:

- **English** (`en`) - **Default language**
- **Hebrew** (`he`) - Full **RTL support**
- Language detection based on **original user-written content**

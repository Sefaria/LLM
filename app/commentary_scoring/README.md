# CommentaryScorer — Commentary–Citation Analysis Tool

**CommentaryScorer** is a Python tool that uses **LLMs** to analyze a **commentary** and determine, for **each cited base text**, whether the commentary actually **explains/interprets** it. It returns a **binary score (0/1)** per citation together with a **short rationale**.

---

## ⭐ Scores Extracted

- **Per-Citation Explanation Score**: `0` = not explained, `1` = explained  
- **Per-Citation Rationale**: short reason string that begins with  
  `Explained spans: '<phrase1>'; '<phrase2>'` (or `'None'`)  

---

## 🚀 Quick Start

```python
from commentary_scoring.commentary_scoring import score_one_commentary
from sefaria_llm_interface.commentary_scoring import CommentaryScoringInput

inp = CommentaryScoringInput(
    commentary_ref="Rashi on Genesis 1:1",
    cited_refs={
        "Genesis 1:1": "In the beginning God created the heavens and the earth.",
        "Genesis 1:2": "Now the earth was formless and void..."
    },
    commentary_text="""
      Rashi on 'בראשית' explains sequencing/purpose and interprets terms...
    """
)

out = score_one_commentary(inp)

print("Scores:", out.ref_scores)
print("Reasons:", out.scores_explanation)

```

## 📦 Data Structures

### **Input — `CommentaryScoringInput`**

```python
{
  "commentary_ref": "Rashi on Genesis 1:1",   # Optional string identifier
  "cited_refs": {                             # Dict of citation → base text
    "Genesis 1:1": "In the beginning ...",
    "Genesis 1:2": "Now the earth ..."
  },
  "commentary_text": "Full commentary text (plain or HTML)"
}
```

- **commentary_ref**: identifier for the commentary (helpful for logging)  
- **cited_refs**: dictionary mapping citation keys (e.g., `"Genesis 1:1"`) to their textual content  
- **commentary_text**: commentary body text (string, can contain HTML, nested lists, etc.)

---

### **Output — `CommentaryScoringOutput`**

```python
{
  "commentary_ref": "Rashi on Genesis 1:1",
  "ref_scores": { "Genesis 1:1": 1, "Genesis 1:2": 0 },
  "scores_explanation": {
    "Genesis 1:1": "Explained spans: 'בראשית'; 'אלוקים' — Adds interpretive content ...",
    "Genesis 1:2": "Explained spans: None — Only a decorative quote, no interpretation ..."
  },
  "processed_datetime": "2025-08-19T10:30:00Z",
  "request_status": 1,
  "request_status_message": ""
}
```

- **ref_scores**: dictionary of binary scores per citation (0 = not explained, 1 = explained)  
- **scores_explanation**: dictionary of rationales per citation, each beginning with **“Explained spans”**  
- **processed_datetime**: UTC ISO8601 timestamp when scoring was done  
- **request_status**: `1 = success`, `0 = failure`  
- **request_status_message**: error description in case of failure

---

## ⚙️ Scoring System

### **Architecture**

The `commentary_scoring` package consists of:

- `commentary_scoring.py` — Main API with `score_one_commentary()`  
- `openai_commentary_scorer.py` — Core LLM engine (`CommentaryScorer`)  
- `tasks.py` — Celery task wrapper for async processing  
- `text_utils.py` — Utilities for HTML stripping and flattening  
- `README.md` — Documentation  


---

### **Explanation Levels**

| Level | Description |
|-------|-------------|
| **0 — Not Explained** | Commentary does not interpret the cited text (decorative prooftext, paraphrase only, inherited interpretation). |
| **1 — Explained**     | Commentary provides interpretation or explanation of any part of the cited text. |

---

## 🧠 Algorithm

### **Input Validation**
- Fail if `cited_refs` is empty or `commentary_text` is missing  
- Token counting via `tiktoken` (fallback = character length)  
- If too long → fail fast with `"input too long"`

### **Build Prompt**
- Commentary text + cited refs in structured sections  
- Explicit instructions for binary labeling per citation  
- Require **“Explained spans”** prefix in explanations

### **Schema Enforcement**
- OpenAI function calling schema requires:  
  - `ref_scores`: dict of citation → 0/1  
  - `explanation`: dict of citation → rationale string  

### **LLM Invocation**
- Config: `gpt-4o-mini`, `temperature=0`, `top_p=0`, `seed=42`  
- Parse structured JSON output

### **Post-Processing**
- Clamp invalid values to `0` or `1`  
- Return `CommentaryScoringOutput`

---

## 🔧 Configuration Options

### **Initialization**

```python
from commentary_scoring.openai_commentary_scorer import CommentaryScorer

scorer = CommentaryScorer(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",        # default model
    max_prompt_tokens=32000,    # max tokens for input prompt
    token_margin=4096           # reserved for model response
)
```

- **API Key**: via `OPENAI_API_KEY` environment variable or explicit parameter  
- **Model**: defaults to `gpt-4o-mini`, override if needed  
- **Token Guardrails**: ensures commentary fits within prompt budget  

---

## 📜 Celery Integration

### **Task Wrapper**

```python
@shared_task(name='llm.score_commentary')
def score_sheet_task(raw_input: dict) -> dict:
    inp = CommentaryScoringInput(**raw_input)
    out = score_one_commentary(inp)
    return asdict(out)
```

### **Usage**

```python
from celery import signature

payload = {
  "commentary_ref": "Rashi on Genesis 1:1",
  "cited_refs": {"Genesis 1:1": "...", "Genesis 1:2": "..."},
  "commentary_text": "Rashi explains ..."
}
sig = signature("llm.score_commentary", args=[payload], queue="llm")
print(sig.apply_async().get())
```

---

## 📊 Output Fields

| Field                  | Description                                      |
|------------------------|--------------------------------------------------|
| `ref_scores`           | Binary 0/1 scores per citation                   |
| `scores_explanation`   | Rationale strings beginning with `"Explained spans"` |
| `commentary_ref`       | Commentary identifier                            |
| `processed_datetime`   | UTC ISO8601 timestamp                            |
| `request_status`       | `1 = success`, `0 = failure`                     |
| `request_status_message` | Error message if failure                        |

---

## 📝 Logging

- **Info**: token count, number of citations, success summary  
- **Warning**: invalid scores clamped, tokenizer fallback  
- **Error**: LLM or JSON parse failures  

```python
import logging
logging.getLogger("commentary_scoring").setLevel(logging.INFO)
```

---

##  ✅ Extensibility

- By now there is no support for very long commentaries, because during testing I didn't encounter any. The chances are high that we won't need this feature at all -- but the matter should be explored.

---


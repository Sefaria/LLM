# Plackett-Luce Active Learning

This experiment ranks items by repeatedly sending small `K`-item groups to an experiment function, observing a full ranking, and updating a Plackett-Luce posterior over item scores.
The main use case here is quoting commentary: instead of asking an LLM to rank every passage at once, we ask it to rank only `K` passages per prompt, collect many partial `K`-way rankings, and then aggregate them into a global ranking with uncertainty estimates.
The active-learning loop then chooses the next `K`-item prompts adaptively, aiming to spend LLM calls where they will reveal the most information about the overall ordering.

## Algorithm

The current implementation has three phases:

1. `warm_start`
   Collect broad random coverage until every item has appeared in at least `warm_start_repeats` experiments.

2. `direct_active`
   Use posterior samples to score candidate `K`-sets and choose one high-value set directly.

3. `mst_batch`
   Compute pairwise mutual-information utilities from posterior samples, build a maximum spanning tree over all items, extract `K`-item neighborhoods, and choose a batch greedily.

Important modeling point:
- the statistical model is full-ranking Plackett-Luce
- each experiment contributes a full `K`-way ranking
- the MST is only a selection heuristic, not part of the probabilistic model

## Main Components

- [active_learning.py](/Users/yon/projects/LLM/experiments/llm_based_sorting/plackett_luce/active_learning.py): active-learning loop and MST batching
- [bayesian_inference.py](/Users/yon/projects/LLM/experiments/llm_based_sorting/plackett_luce/bayesian_inference.py): Gibbs sampler for Plackett-Luce posterior inference
- [experiment_runners.py](/Users/yon/projects/LLM/experiments/llm_based_sorting/plackett_luce/experiment_runners.py): synthetic and Claude-based ranking runners
- [sefaria_retrieval.py](/Users/yon/projects/LLM/experiments/llm_based_sorting/plackett_luce/sefaria_retrieval.py): quoting-commentary retrieval from Sefaria

## Visualization 1: Synthetic Demo

This demo uses hidden synthetic scores and a mock experiment runner.

Run:

```bash
streamlit run experiments/llm_based_sorting/plackett_luce/streamlit_demo.py
```

Optional custom port:

```bash
streamlit run experiments/llm_based_sorting/plackett_luce/streamlit_demo.py --server.port 8502
```

What you see:
- posterior mean and uncertainty
- pairwise mutual-information heatmap
- maximum spanning tree
- selected candidate `K`-sets
- observation history

## Visualization 2: Sefaria Quoting Commentary Demo

This demo retrieves quoting-commentary passages for a Sefaria ref and uses the active-learning loop with a Claude ranking runner.

Run:

```bash
streamlit run experiments/llm_based_sorting/plackett_luce/streamlit_sefaria_demo.py
```

Optional custom port:

```bash
streamlit run experiments/llm_based_sorting/plackett_luce/streamlit_sefaria_demo.py --server.port 8503
```

What you can control in the UI:
- Sefaria ref
- relevance prompt
- `K` items per LLM prompt
- warm-start repeats
- direct-active rounds
- MST batch size
- number of iterations to run per click
- max loaded items and sampling seed for smaller demos

What you see:
- retrieved item map from item id to commentary passage
- posterior summary
- pairwise mutual-information heatmap
- maximum spanning tree
- selected candidate sets
- observation history
- actual commentary texts
- total LLM call count

Notes:
- the display language in the Sefaria demo affects the UI text display
- the text sent to Claude always prefers English and falls back to Hebrew
- each experiment call sends exactly `K` passages to the LLM and expects a full ranking back

## Environment Notes

The Sefaria demo assumes:
- a working Python environment with the required dependencies
- access to the sibling `Sefaria-Project` checkout when using `local_project` retrieval
- `ANTHROPIC_API_KEY` set when using the Claude ranking runner

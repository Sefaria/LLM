"""
A script to run the LLM and CNN linker and then compare the results.
"""
from experiments.linker.fine_tune.project_scripts.run_on_validation_set import run_llm_linker
from experiments.linker.fine_tune.project_scripts.run_cnn_linker import run_linker_on_collection
from experiments.linker.fine_tune.project_scripts.diff_prodigy_collections import diff_prodigy_collections


if __name__ == '__main__':
    # run_llm_linker("ner_he_input_broken", "ner_he_cnn_copper")
    # run_linker_on_collection('ner_he_input_broken', 'ner_he_cnn_copper', 'he')
    diff_prodigy_collections('ner_he_cnn_copper', 'ner_he_gpt_copper', 'ner_he_diff')

"""
Given clusters tries to
1) Choose the "best" clusters such that each cluster represents an idea that should be part of the final topic page curation
2) For each chosen cluster, choose the "best" source to be curated. Best here needs to fulfill a few criteria including
    - Category quota: sources should be from diverse categories in Sefaria
    - Fundamental sources: funadmental sources from Tanakh and Talmud etc. should be chosen. These should be few, made 2-3
    - Interesting sources: the rest of the sources should represent interesting ideas for a newcomer to Sefaria
"""
from tqdm import tqdm
from experiments.topic_source_curation_v2.cluster import Cluster, SummarizedSource, embed_text_openai, get_text_from_source
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sklearn.metrics import pairwise_distances
from util.pipeline import Artifact
import numpy as np
from basic_langchain.schema import HumanMessage, SystemMessage
from basic_langchain.chat_models import ChatOpenAI
from sefaria_llm_interface.common.topic import Topic
# import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, LpSolverDefault
from sefaria.model.text import Ref
from functools import reduce
from sefaria.model import library



def solve_clusters(clusters: list[Cluster], sorted_items: list[SummarizedSource], primary_trefs: list[str]) -> (list[SummarizedSource], list[str]):
    num_of_sources = sum(len(c.items) for c in clusters)
    flattened_summarized_sources = reduce(lambda x, y: x + y.items, clusters, [])
    prob = LpProblem("Choose_Sources", LpMaximize)

    ref_var_map = {}
    var_ref_map = {}
    ordered_sources_vars = []
    # for cluster in clusters:
    for item in sorted_items:
        var = LpVariable(f'{item.source.ref}', lowBound=0, upBound=1, cat='Binary')
        ref_var_map[item.source.ref] = var
        var_ref_map[var] = item.source.ref
        ordered_sources_vars.append(var)


    # Add the constraints that sum of all sources from the same cluster must be equal to 1
    for cluster in clusters:
        cluster_vars = [ref_var_map[item.source.ref] for item in cluster.items]
        prob += lpSum(cluster_vars) <= 1


    # Add the constraints that the sum of sources with the same category must be >= 1
    category_vars_map = {}
    for ref, var in ref_var_map.items():
        category = Ref(ref).primary_category
        if category not in category_vars_map:
            category_vars_map[category] = []
        category_vars_map[category].append(var)

    missing_category_penalty_vars = []
    for category, vars_in_category in category_vars_map.items():
        missing_category_penalty_var = LpVariable(f'penalty_missing_category_{category}', lowBound=0, upBound=1, cat='Binary')
        prob += lpSum(vars_in_category) + missing_category_penalty_var >= 1
        missing_category_penalty_vars.append(missing_category_penalty_var)

    # primary sources must be chosen
    primary_vars = [ref_var_map[primary_ref] for primary_ref in primary_trefs]
    for primary_sources_var in primary_vars:
        prob += primary_sources_var == 1

    # don't choose sources from the same book
    book_vars_map = {}
    for ref, var in ref_var_map.items():
        book = Ref(ref).index.title
        if book not in book_vars_map:
            book_vars_map[book] = []
        book_vars_map[book].append(var)
    same_book_penalty_vars = []
    for book, vars in book_vars_map.items():
        same_book_penalty_var = LpVariable(f'penalty_same_book_{book}', lowBound=0, cat='Integer')
        prob += lpSum(vars) - same_book_penalty_var <= 1
        same_book_penalty_vars.append(same_book_penalty_var)

    # don't choose sources from the same author
    author_vars_map = {}
    for ref, var in ref_var_map.items():
        book = Ref(ref).index.title
        authors = getattr(library.get_index(book), 'authors', [])
        for author in authors:
            if author not in author_vars_map:
                author_vars_map[author] = []
            author_vars_map[author].append(var)
    same_author_penalty_vars = []
    for author, vars in author_vars_map.items():
        if len(vars) < 2:
            continue
        same_author_penalty_var = LpVariable(f'penalty_same_author_{author}', lowBound=0, cat='Integer')
        prob += lpSum(vars) - same_book_penalty_var <= 1
        same_author_penalty_vars.append(same_author_penalty_var)


    # Add constraint that number of chosen sources should not exceed 20
    prob += lpSum(ordered_sources_vars) == min(20, len(clusters))

    weights = [num_of_sources - i for i in range(len(ordered_sources_vars))]
    weighted_sources = [weights[i] * ordered_sources_vars[i] for i in range(len(ordered_sources_vars))]
    mean_weight = int(sum(weights) / len(weights))
    objective_function = lpSum(weighted_sources) - num_of_sources*lpSum(missing_category_penalty_vars) - 5*num_of_sources*lpSum(same_book_penalty_vars) -num_of_sources*(same_author_penalty_vars)
    prob += objective_function

    # Solve the problem
    LpSolverDefault.msg = 1
    prob.solve()

    # Print the results
    print("Status:", LpStatus[prob.status])
    print("Selected sources:")
    for var in ordered_sources_vars:
        if var.varValue > 0:
            ref = var_ref_map[var]
            print(f"https://www.sefaria.org/{Ref(ref).url()} value={var.varValue}")

    print("Penalties:")
    chosen_penalties = []
    for penalty in missing_category_penalty_vars:
        if penalty.varValue and penalty.varValue > 0:
            print(f"{penalty.name} = {penalty.varValue}")
            chosen_penalties.append(penalty.name)
    for penalty in same_book_penalty_vars:
        if penalty.varValue and penalty.varValue > 0:
            print(f"{penalty.name} = {penalty.varValue}")
            chosen_penalties.append(penalty.name)

    chosen_sources = [item for item in flattened_summarized_sources if ref_var_map[item.source.ref].varValue > 0]

    return chosen_sources, chosen_penalties






if __name__ == "__main__":
    print("HI")

    # Create a problem variable
    prob = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

    # Define decision variables
    x = pulp.LpVariable('x', lowBound=0, cat='Integer')  # Number of units of Product A
    y = pulp.LpVariable('y', lowBound=0, cat='Integer')  # Number of units of Product B

    # Objective function
    prob += 20 * x + 30 * y, "Total Profit"

    # Constraints
    prob += 2 * x + y <= 40, "Machine_1_Time_Constraint"
    prob += x + y <= 30, "Machine_2_Time_Constraint"

    # Solve the problem
    prob.solve()

    # Print the results
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Optimal number of units of Product A to produce: {x.varValue}")
    print(f"Optimal number of units of Product B to produce: {y.varValue}")
    print(f"Maximum Profit: ${pulp.value(prob.objective)}")


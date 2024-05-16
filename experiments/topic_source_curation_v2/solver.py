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
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from sefaria.model.text import Ref
from functools import reduce



def solve_clusters(clusters: list[Cluster]) -> list[SummarizedSource]:
    num_of_sources = sum(len(c.items) for c in clusters)
    flattened_summarized_sources = reduce(lambda x, y: x + y.items, clusters, [])
    prob = LpProblem("Choose_Sources", LpMaximize)
    source_vars = []
    category_dict = {}
    for cluster in clusters:
        cluster_vars = []
        for item in cluster.items:
            var = LpVariable(f'{item.source.ref}', lowBound=0, upBound=1, cat='Binary')
            cluster_vars.append(var)
            # Organize by category
            category = Ref(item.source.ref).primary_category
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(var)
        source_vars.append(cluster_vars)

    # Add the constraints that sum of all sources from the same cluster must be equal to 1
    for cluster_vars in source_vars:
        prob += lpSum(cluster_vars) == 1

    # Add the constraints that the sum of sources with the same category must be >= 1
    for category, vars_in_category in category_dict.items():
        prob += lpSum(vars_in_category) >= 1

    all_vars = [var for inner_list in source_vars for var in inner_list]

    # # Ad constraint that number of chosen sources should not exceed 20
    # prob += lpSum(all_vars) == 20

    weight = num_of_sources
    objective_function = lpSum([(weight - i) * var for i, var in enumerate(all_vars)])

    prob += objective_function
    prob.solve()
    print(f"Status: {LpStatus[prob.status]}")
    print("Selected Sources:")

    curated_sources = []
    sorted_vars = sorted(all_vars, key=lambda var: var.varValue, reverse=True)
    for var in sorted_vars:
        # if var.varValue < 0:
        ref = var.name.replace("_", " ")
        print(f"Source: {ref}", f"Value: {var.varValue}")
        ref = var.name.replace("_", " ")
        # source = [source for source in flattened_summarized_sources if source.]
    print("hi")





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


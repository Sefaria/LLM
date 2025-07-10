import django
django.setup()
from sefaria.model import IntraTopicLink, IntraTopicLinkSet
from sefaria.pagesheetrank import pagerank






if __name__ == '__main__':
    # Get all intra-topic links
    intra_topic_links = IntraTopicLinkSet().array()
    # for link in intra_topic_links:
    #     print(f"{link.fromTopic} -> {link.toTopic}")
    edges = [
        # (link.fromTopic, link.toTopic)  # ordered pair
        (link.toTopic, link.fromTopic)
        for link in IntraTopicLinkSet().array()  # iterate through all links
    ]
    # Print the edges
    for edge in edges:
        print(f"{edge[0]} -> {edge[1]}")
    # edges = list(set(edges))

    # 2. accumulate them into the dict-of-dicts shape
    graph = {}
    for src, dst in edges:
        graph.setdefault(src, {})
        graph[src][dst] = graph[src].get(dst, 0) + 1  # count multiple links

    # 3. make sure *every* slug appears, even if it never links out
    for src, dst in edges:
        graph.setdefault(dst, {})

    pr_scores = pagerank(list(graph.items()))
    sorted_scores = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    for slug, score in sorted_scores[:100]:
        # print(f"{slug:20} {score:.6f}")
        print(f"{slug}")

    index = None
    for i , (slug, score) in enumerate(sorted_scores):
        if slug == "life":
            index = i
            break
    print(f"Index of 'life': {index}")



    # # Filter out overly generalized topics
    # overly_generalized_topics = [
    #     link for link in intra_topic_links if len(link.slugs) > 10
    # ]
    #
    # # Print the overly generalized topics
    # for topic in overly_generalized_topics:
    #     print(f"Topic: {topic.ref}, Slugs: {topic.slugs}")
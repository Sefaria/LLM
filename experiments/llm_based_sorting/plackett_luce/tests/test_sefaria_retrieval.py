from __future__ import annotations

from experiments.llm_based_sorting.plackett_luce.sefaria_retrieval import (
    LocalSefariaProjectQuotingCommentaryRetriever,
    SefariaApiQuotingCommentaryRetriever,
    build_item_text_lookup_from_links,
)


def test_retriever_builds_expected_api_url_and_normalizes_response() -> None:
    seen_urls: list[str] = []

    def fake_fetcher(url: str):
        seen_urls.append(url)
        return [
            {
                "anchorRef": "Genesis 1:1",
                "sourceRef": "Rashi on Genesis 1:1:1",
                "category": "Quoting Commentary",
                "index_title": "Rashi on Genesis",
                "collectiveTitle": {"en": "Rashi", "he": "רש\"י"},
                "text": ["In the beginning..."],
                "he": ["בראשית..."],
            }
        ]

    retriever = SefariaApiQuotingCommentaryRetriever(
        base_url="http://localhost:8000",
        json_fetcher=fake_fetcher,
    )
    items = retriever.fetch("Genesis 1:1")

    assert seen_urls == [
        "http://localhost:8000/api/links/Genesis%201%3A1?with_text=1&categories=Quoting%20Commentary"
    ]
    assert len(items) == 1
    assert items[0].source_ref == "Rashi on Genesis 1:1:1"
    assert items[0].render("en") == "In the beginning..."


def test_build_item_text_lookup_includes_metadata_and_text() -> None:
    retriever = SefariaApiQuotingCommentaryRetriever(
        json_fetcher=lambda _: [
            {
                "anchorRef": "Genesis 1:1",
                "sourceRef": "Rashi on Genesis 1:1:1",
                "category": "Quoting Commentary",
                "index_title": "Rashi on Genesis",
                "collectiveTitle": {"en": "Rashi", "he": "רש\"י"},
                "text": ["Line one", "Line two"],
                "he": ["שורה א", "שורה ב"],
            }
        ]
    )

    items = retriever.fetch("Genesis 1:1")
    lookup = build_item_text_lookup_from_links(items, language="en")

    assert "Source ref: Rashi on Genesis 1:1:1" in lookup[0]
    assert "Anchor ref: Genesis 1:1" in lookup[0]
    assert "Line one\nLine two" in lookup[0]


def test_local_project_retriever_uses_get_links_callable() -> None:
    retriever = LocalSefariaProjectQuotingCommentaryRetriever(
        get_links_func=lambda tref, with_text, categories: [
            {
                "anchorRef": tref,
                "sourceRef": "Ibn Ezra on Genesis 1:1:1",
                "category": "Quoting Commentary",
                "index_title": "Ibn Ezra on Genesis",
                "collectiveTitle": {"en": "Ibn Ezra", "he": "אבן עזרא"},
                "text": ["example"],
                "he": ["דוגמה"],
            }
        ]
    )

    items = retriever.fetch("Genesis 1:1")

    assert len(items) == 1
    assert items[0].collective_title_en == "Ibn Ezra"

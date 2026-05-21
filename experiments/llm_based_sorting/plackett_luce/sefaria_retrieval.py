from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib.parse import quote
from urllib.request import urlopen


JsonDict = dict[str, Any]


class JsonFetcher(Protocol):
    def __call__(self, url: str) -> Any:
        ...


def default_json_fetcher(url: str) -> Any:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class QuotingCommentaryItem:
    anchor_ref: str
    source_ref: str
    category: str
    index_title: str
    collective_title_en: str
    collective_title_he: str
    text: list[str] | str | None
    he: list[str] | str | None
    raw_link: JsonDict

    def render(self, language: str = "en") -> str:
        payload = self.text if language == "en" else self.he
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        return "\n".join(str(segment) for segment in payload)

    def render_with_fallback(self, language: str = "en") -> tuple[str, str]:
        primary_text = self.render(language=language).strip()
        if primary_text:
            return primary_text, language

        fallback_language = "he" if language == "en" else "en"
        fallback_text = self.render(language=fallback_language).strip()
        if fallback_text:
            return fallback_text, fallback_language

        return "", language


class SefariaApiQuotingCommentaryRetriever:
    """
    Retrieve quoting-commentary links using the same link objects that power the
    Sefaria client.

    Relevant upstream behavior:
    - the client loads related links for a section via `Sefaria.related(...)`
    - quoting commentary is identified client-side by `category === "Quoting Commentary"`
    - the server can already return text-enriched link objects through
      `/api/links/<ref>?with_text=1`

    For ranking experiments, the most direct API call is:
        /api/links/<ref>?with_text=1&categories=Quoting%20Commentary
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        json_fetcher: JsonFetcher = default_json_fetcher,
    ):
        self.base_url = base_url.rstrip("/")
        self.json_fetcher = json_fetcher

    def fetch(self, tref: str) -> list[QuotingCommentaryItem]:
        encoded_ref = quote(tref, safe="")
        url = (
            f"{self.base_url}/api/links/{encoded_ref}"
            "?with_text=1&categories=Quoting%20Commentary"
        )
        payload = self.json_fetcher(url)
        if not isinstance(payload, list):
            raise ValueError("Expected a list response from the Sefaria links API.")
        return [self._normalize_link(link) for link in payload]

    @staticmethod
    def _normalize_link(link: JsonDict) -> QuotingCommentaryItem:
        collective_title = link.get("collectiveTitle", {})
        return QuotingCommentaryItem(
            anchor_ref=link["anchorRef"],
            source_ref=link["sourceRef"],
            category=link["category"],
            index_title=link["index_title"],
            collective_title_en=collective_title.get("en", ""),
            collective_title_he=collective_title.get("he", ""),
            text=link.get("text"),
            he=link.get("he"),
            raw_link=link,
        )


class LocalSefariaProjectQuotingCommentaryRetriever:
    """
    Retrieve quoting commentary directly from a local sibling Sefaria-Project
    checkout, without requiring the HTTP API server to be running.
    """

    def __init__(
        self,
        *,
        sefaria_project_path: str = "/Users/yon/projects/Sefaria-Project",
        get_links_func=None,
    ):
        self.sefaria_project_path = sefaria_project_path
        self._get_links_func = get_links_func

    def fetch(self, tref: str) -> list[QuotingCommentaryItem]:
        get_links = self._get_links_func or self._load_get_links()
        payload = get_links(tref, with_text=True, categories=["Quoting Commentary"])
        if not isinstance(payload, list):
            raise ValueError("Expected a list response from local Sefaria get_links().")
        return [SefariaApiQuotingCommentaryRetriever._normalize_link(link) for link in payload]

    def _load_get_links(self):
        bootstrap_sefaria(self.sefaria_project_path)
        try:
            from sefaria.client.wrapper import get_links
        except ModuleNotFoundError as exc:
            if exc.name == "django_recaptcha":
                raise ModuleNotFoundError(
                    "Sefaria bootstrap failed because the active Python environment is missing "
                    "`django-recaptcha`, which Sefaria-Project requires during django.setup(). "
                    "Use the usual Sefaria environment or install that dependency there."
                ) from exc
            raise
        return get_links


def bootstrap_sefaria(sefaria_project_path: str) -> None:
    project_path = os.path.abspath(sefaria_project_path)
    if not os.path.isdir(project_path):
        raise FileNotFoundError(f"Sefaria-Project path does not exist: {project_path}")

    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sefaria.settings")

    import django

    django.setup()

    from sefaria.model.text import library

    # `get_links()` expects the library TOC cache to exist. In Sefaria's normal
    # startup path this is populated by reader.startup.init_library_cache().
    # For this lightweight script bootstrap, initializing the TOC tree is
    # enough to make `library.get_collections_in_library()` safe.
    if getattr(library, "_toc_tree", None) is None:
        library.get_toc_tree()


def build_item_text_lookup_from_links(
    items: Sequence[QuotingCommentaryItem],
    *,
    language: str = "en",
    include_metadata: bool = True,
) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for item_id, item in enumerate(items):
        body = item.render(language=language).strip()
        if include_metadata:
            text = "\n".join(
                [
                    f"Source ref: {item.source_ref}",
                    f"Anchor ref: {item.anchor_ref}",
                    f"Title: {item.collective_title_en or item.index_title}",
                    "Text:",
                    body,
                ]
            )
        else:
            text = body
        lookup[item_id] = text
    return lookup

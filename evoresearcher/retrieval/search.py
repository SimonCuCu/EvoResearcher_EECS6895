"""Web search and page extraction."""

from __future__ import annotations

from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from bs4 import BeautifulSoup
import httpx

from evoresearcher.schemas import SourceNote


class WebResearcher:
    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                )
            },
            follow_redirects=True,
        )

    def search(self, query: str, limit: int = 3) -> list[SourceNote]:
        results = self._search_html(query, limit=limit)
        if results:
            return results[:limit]
        return self._search_lite(query, limit=limit)

    def _clean_result_url(self, url_value: str) -> str:
        if url_value.startswith("//"):
            url_value = f"https:{url_value}"
        elif url_value.startswith("/"):
            url_value = f"https://duckduckgo.com{url_value}"
        parsed = urlparse(url_value)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
        return url_value

    def _search_html(self, query: str, *, limit: int) -> list[SourceNote]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SourceNote] = []
        for block in soup.select(".result"):
            title_node = block.select_one(".result__title")
            link_node = block.select_one(".result__url")
            snippet_node = block.select_one(".result__snippet")
            if title_node is None or link_node is None:
                continue
            href = block.select_one(".result__title a")
            url_value = href.get("href", "").strip() if href else ""
            url_value = self._clean_result_url(url_value)
            if not url_value.startswith("http") or self._is_ad_url(url_value):
                continue
            results.append(
                SourceNote(
                    title=title_node.get_text(" ", strip=True),
                    url=url_value,
                    snippet="" if snippet_node is None else snippet_node.get_text(" ", strip=True),
                )
            )
            if len(results) >= limit:
                break
        return results

    def _search_lite(self, query: str, *, limit: int) -> list[SourceNote]:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        response = self.client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SourceNote] = []
        for link in soup.select("a.result-link"):
            url_value = self._clean_result_url(link.get("href", "").strip())
            if not url_value.startswith("http") or self._is_ad_url(url_value):
                continue
            row = link.find_parent("tr")
            snippet = ""
            if row is not None:
                next_row = row.find_next_sibling("tr")
                if next_row is not None:
                    snippet = next_row.get_text(" ", strip=True)
            results.append(
                SourceNote(
                    title=link.get_text(" ", strip=True),
                    url=url_value,
                    snippet=snippet,
                )
            )
            if len(results) >= limit:
                break
        return results

    def _is_ad_url(self, url_value: str) -> bool:
        parsed = urlparse(url_value)
        return parsed.netloc.endswith("duckduckgo.com") and parsed.path.endswith("/y.js")

    def enrich(self, source: SourceNote, char_limit: int = 1600) -> SourceNote:
        try:
            response = self.client.get(source.url)
            response.raise_for_status()
        except Exception:
            return source
        soup = BeautifulSoup(response.text, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.extract()
        text = " ".join(soup.get_text(" ", strip=True).split())
        return source.model_copy(update={"excerpt": text[:char_limit]})

from evoresearcher.retrieval.search import WebResearcher


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSearchClient:
    def __init__(self):
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        if "html.duckduckgo.com" in url:
            return FakeResponse("<html><body>No result markup</body></html>")
        return FakeResponse(
            """
            <html><body><table>
              <tr><td><a class="result-link"
                href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpaper%3Fx%3D1">
                Example paper
              </a></td></tr>
              <tr><td class="result-snippet">Useful snippet.</td></tr>
            </table></body></html>
            """
        )


def test_search_falls_back_to_lite_and_unwraps_duckduckgo_redirects():
    researcher = WebResearcher()
    researcher.client = FakeSearchClient()

    results = researcher.search("social protection", limit=1)

    assert len(results) == 1
    assert results[0].title == "Example paper"
    assert results[0].url == "https://example.org/paper?x=1"
    assert "Useful snippet" in results[0].snippet
    assert any("html.duckduckgo.com" in url for url in researcher.client.urls)
    assert any("lite.duckduckgo.com" in url for url in researcher.client.urls)


def test_clean_result_url_unwraps_relative_duckduckgo_redirect():
    researcher = WebResearcher()

    cleaned = researcher._clean_result_url(
        "/l/?uddg=https%3A%2F%2Fexample.org%2Frelative%3Fx%3D1"
    )

    assert cleaned == "https://example.org/relative?x=1"

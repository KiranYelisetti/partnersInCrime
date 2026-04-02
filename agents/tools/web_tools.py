"""
Web tools for agents — search the internet and fetch page content.

This solves the stale-knowledge problem: local LLMs have training cutoffs,
so they might not know the latest API changes, library versions, or best practices.
Agents can now Google things mid-task, just like a human dev would.

Uses DuckDuckGo (free, no API key) and BeautifulSoup for page extraction.
"""
import requests
from langchain_core.tools import tool

from config import COMMAND_TIMEOUT


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo. Use this to find latest docs, API references,
    library versions, or solutions to errors you encounter.

    Args:
        query: What to search for (e.g. "FastAPI 0.115 middleware changes")
        max_results: Number of results to return (default 5, max 10)
    Returns:
        Search results with title, URL, and snippet for each result.
    """
    max_results = min(max_results, 10)

    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", r.get("link", ""))
            snippet = r.get("body", r.get("snippet", ""))
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Search error: {e}"


@tool
def web_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch a web page and extract its text content. Use this to read documentation,
    API references, or Stack Overflow answers.

    Args:
        url: The URL to fetch (e.g. a docs page or API reference)
        max_chars: Max characters to return (default 5000, max 10000)
    Returns:
        The extracted text content of the page.
    """
    max_chars = min(max_chars, 10000)

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PartnersInCrime/1.0; +research-agent)"
        }
        resp = requests.get(url, headers=headers, timeout=COMMAND_TIMEOUT)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")

        # If it's JSON, return it directly (API docs often serve JSON)
        if "json" in content_type:
            text = resp.text[:max_chars]
            return f"JSON from {url}:\n{text}"

        # Parse HTML and extract text
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "iframe", "noscript"]):
            tag.decompose()

        # Try to find the main content area first
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find("div", class_="content")
            or soup.find("div", class_="documentation")
            or soup.body
            or soup
        )

        text = main.get_text(separator="\n", strip=True)

        # Clean up excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text) - max_chars} chars remaining)"

        if not text.strip():
            return f"Page at {url} returned no extractable text content."

        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out fetching {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    except Exception as e:
        return f"Error processing {url}: {e}"


# Export
WEB_TOOLS = [web_search, web_fetch]

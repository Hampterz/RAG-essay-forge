"""
Faculty search using the Semantic Scholar Graph API (free, official, no key
required for light use — https://api.semanticscholar.org/api-docs/graph).

Semantic Scholar doesn't have a "search by university department" endpoint,
so the approach is:
  1. Search authors by a query that mixes topic + school name
     (e.g. "robotics UC San Diego") — Semantic Scholar's author search
     matches against name AND recent paper text/venues reasonably well.
  2. Pull each matching author's recent papers to build a "research
     interests" summary from paper titles/abstracts.
  3. Affiliation comes from the author record when available.

This will not be as clean as a curated faculty directory, but it's free,
live, and requires zero scraping or maintenance.
"""

import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"


def search_authors(query, limit=10):
    """
    query: free text like "machine learning robotics UC San Diego"
    Returns a list of dicts: name, affiliations, paper_count, authorId
    """
    resp = requests.get(
        f"{BASE_URL}/author/search",
        params={
            "query": query,
            "fields": "name,affiliations,paperCount,homepage,url",
            "limit": limit,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_author_detail(author_id):
    """
    Fetch an author's recent papers to derive research interests + a
    best-guess contact email (Semantic Scholar does NOT provide emails —
    see note in app.py / README about filling this in manually or via
    the university directory).
    """
    resp = requests.get(
        f"{BASE_URL}/author/{author_id}",
        params={
            "fields": "name,affiliations,homepage,papers.title,papers.abstract,papers.year,papers.venue",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    papers = data.get("papers", []) or []
    papers_sorted = sorted(papers, key=lambda p: p.get("year") or 0, reverse=True)
    top_papers = papers_sorted[:5]

    titles = [p["title"] for p in top_papers if p.get("title")]
    research_summary = "; ".join(titles) if titles else "No recent papers found"

    return {
        "name": data.get("name"),
        "affiliations": data.get("affiliations", []),
        "homepage": data.get("homepage"),
        "research_summary": research_summary,
        "top_paper_titles": titles,
    }

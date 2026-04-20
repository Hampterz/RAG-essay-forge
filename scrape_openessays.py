#!/usr/bin/env python3
"""
Scrape publicly listed college admission essays from openessays.org.

Listing: URLs are taken from the site's sitemap (reliable; homepage HTML embeds
essay JSON but server-side listings do not vary with ?page= for simple GETs).

Essay pages: https://www.openessays.org/essays/{slug}
Full text is in <main> under div.prose (Tailwind prose class).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE = "https://www.openessays.org"
SITEMAP_URL = f"{BASE}/sitemap.xml"
USER_AGENT = (
    "OpenEssaysScraper/1.0 (+https://www.openessays.org/; "
    "educational research; respects crawl-delay)"
)
REQUEST_TIMEOUT = 60


def polite_sleep(min_s: float = 2.0, max_s: float = 3.0) -> None:
    time.sleep(random.uniform(min_s, max_s))


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _canonical_www(url: str) -> str:
    """Sitemap uses openessays.org; normalize to www to avoid extra redirects."""
    url = url.strip()
    for prefix in ("http://openessays.org", "https://openessays.org"):
        if url.startswith(prefix):
            return "https://www.openessays.org" + url[len(prefix) :]
    return url


def discover_essay_urls(sess: requests.Session) -> list[str]:
    """Return essay page URLs from sitemap.xml."""
    r = sess.get(SITEMAP_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []
    for u in root.findall("sm:url", ns):
        loc = u.find("sm:loc", ns)
        if loc is None or not loc.text:
            continue
        url = _canonical_www(loc.text)
        if "/essays/" in url:
            urls.append(url)
    # Stable order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _split_h1(h1: str) -> tuple[str, str]:
    """
    Headings look like: 'Common App EssayEssay -Harvard University'
    or 'Statement of PurposeEssay -Carnegie Mellon University'
    """
    if "Essay -" in h1:
        left, right = h1.split("Essay -", 1)
        return left.strip(), right.strip()
    return "", h1.strip()


def _metadata_from_main(main: BeautifulSoup) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    row = main.select_one("div.flex.flex-wrap.gap-4.text-sm")
    if row:
        for pill in row.select("div[class*='rounded-full']"):
            text = pill.get_text(strip=True)
            if ":" in text:
                key, val = text.split(":", 1)
                meta[key.strip()] = val.strip()

    for line in main.select("div.flex.items-center.gap-2.text-sm"):
        t = line.get_text(strip=True)
        if t.startswith("Source:"):
            rest = t.split(":", 1)[1].strip() if ":" in t else t
            rest = rest.replace("View Original", "").strip()
            meta["Source"] = rest

    for a in main.find_all("a", href=True):
        if "View Original" in a.get_text():
            meta["original_url"] = a["href"]
            break

    return meta


def parse_essay_page(html: str, page_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        raise ValueError("No <main> element")

    h1 = main.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""
    essay_type, school = _split_h1(h1_text)

    prose = main.select_one("div.prose")
    essay_text = prose.get_text("\n", strip=True) if prose else ""

    extra = _metadata_from_main(main)
    slug_match = re.search(r"/essays/([^/?#]+)", page_url)
    meta_out: dict[str, Any] = {
        "page_url": page_url,
        "slug": slug_match.group(1) if slug_match else "",
        "program": extra.get("Program", ""),
        "program_type": extra.get("Type", ""),
        "license": extra.get("License", ""),
        "source_attribution": extra.get("Source", ""),
        "original_url": extra.get("original_url", ""),
    }
    return {
        "school": school,
        "essay_type": essay_type,
        "essay_text": essay_text,
        "metadata": json.dumps(meta_out, ensure_ascii=False),
    }


def scrape(
    out_csv: str,
    max_essays: int | None = None,
    dry_run: bool = False,
) -> None:
    log = logging.getLogger(__name__)
    sess = session()

    log.info("Fetching sitemap: %s", SITEMAP_URL)
    urls = discover_essay_urls(sess)
    log.info("Found %s essay URLs in sitemap", len(urls))
    if max_essays is not None:
        urls = urls[: max(0, max_essays)]

    if dry_run:
        for u in urls[:10]:
            print(u)
        print("...", f"(showing first 10 of {len(urls)})")
        return

    fieldnames = ["school", "essay_type", "essay_text", "metadata"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, url in enumerate(urls):
            try:
                if i > 0:
                    polite_sleep()
                r = sess.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                row = parse_essay_page(r.text, url)
                writer.writerow(row)
                f.flush()
                log.info("[%s/%s] OK %s", i + 1, len(urls), url)
            except Exception as e:
                log.warning("Skip %s: %s", url, e)
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape OpenEssays.org essays to CSV.")
    parser.add_argument(
        "-o",
        "--output",
        default="openessays_essays.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N essays (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list URLs from sitemap, no essay requests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    scrape(args.output, max_essays=args.max, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

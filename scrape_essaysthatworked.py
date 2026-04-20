#!/usr/bin/env python3
"""
Scrape essay examples from essaysthatworked.com (essay library with grade filters).

Discovery (from inspecting the site):
- The listing page HTML embeds __NEXT_DATA__ with essays, but pagination in that
  blob does not match the client-rendered list for different ?page= values.
- The browser loads paginated data from the public API:
    GET https://api.essaysthatworked.com/v1/essayLibrary/browse?page=N&grades=0,1
  (grades=0 → A+ tier, 1 → A tier in the UI filter you used.)
- Full essay body HTML is only complete on each essay page:
    https://essaysthatworked.com/essay-library/{id}
  (also reachable via redirect from /essays/{id})

This script uses requests + BeautifulSoup only (no Playwright). If the API shape
changes, you may need to inspect network requests again.
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
from typing import Any

import requests
from bs4 import BeautifulSoup

SITE = "https://essaysthatworked.com"
BROWSE_API = "https://api.essaysthatworked.com/v1/essayLibrary/browse"
USER_AGENT = (
    "ETWScraper/1.0 (+https://essaysthatworked.com/essay-library; "
    "research; polite rate limit)"
)
REQUEST_TIMEOUT = 60

TYPE_LABELS: dict[str, str] = {
    "personal_statement": "Personal Statement",
    "supplemental": "Supplemental",
    "university_of_california": "University of California (PIQ)",
}


def polite_sleep(min_s: float = 2.0, max_s: float = 3.0) -> None:
    time.sleep(random.uniform(min_s, max_s))


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def fetch_browse_page(
    sess: requests.Session, page: int, grades: str
) -> dict[str, Any]:
    r = sess.get(
        BROWSE_API,
        params={"page": page, "grades": grades},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def iter_browse_essays(
    sess: requests.Session,
    grades: str,
    max_summaries: int | None = None,
) -> list[dict[str, Any]]:
    """Return essay summary dicts from the browse API (all pages unless capped)."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        if page > 1:
            polite_sleep()
        data = fetch_browse_page(sess, page, grades)
        if not data.get("success"):
            logging.warning("browse page %s: success=false: %s", page, data)
            break
        essays = data.get("essays") or []
        out.extend(essays)
        if max_summaries is not None and len(out) >= max_summaries:
            return out[:max_summaries]
        pag = data.get("pagination") or {}
        max_page = int(pag.get("max", page))
        if page >= max_page or not essays:
            break
        page += 1
    return out


def parse_next_data(html: str) -> dict[str, Any]:
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        html,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Missing __NEXT_DATA__")
    return json.loads(m.group(1))


def html_to_plain(fragment: str | None) -> str:
    if not fragment:
        return ""
    return BeautifulSoup(fragment, "html.parser").get_text("\n", strip=True)


def build_essay_type(essay: dict[str, Any]) -> str:
    raw = essay.get("type") or ""
    label = TYPE_LABELS.get(raw, raw.replace("_", " ").title())
    prompt = essay.get("prompt") or {}
    prompt_plain = html_to_plain(prompt.get("html"))
    if prompt_plain:
        return f"{label} — {prompt_plain}"
    return label


def fetch_essay_detail(sess: requests.Session, essay_id: int) -> dict[str, Any]:
    url = f"{SITE}/essay-library/{essay_id}"
    r = sess.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"Referer": f"{SITE}/essay-library"},
    )
    r.raise_for_status()
    data = parse_next_data(r.text)
    essay = data.get("props", {}).get("pageProps", {}).get("essay")
    if not essay:
        raise ValueError("No essay in pageProps")
    return essay


def row_from_essay(sess: requests.Session, summary: dict[str, Any]) -> dict[str, Any]:
    eid = int(summary["id"])
    detail = fetch_essay_detail(sess, eid)

    schools = detail.get("schools") or []
    school_names = [s.get("name", "") for s in schools if s.get("name")]
    school = school_names[0] if school_names else ""

    grade = detail.get("grade") or summary.get("grade") or ""
    essay_type = build_essay_type(detail)
    essay_text = html_to_plain(detail.get("html"))

    meta: dict[str, Any] = {
        "essay_id": eid,
        "title": detail.get("title"),
        "wordCount": detail.get("wordCount"),
        "viewCount": detail.get("viewCount"),
        "createdAt": detail.get("createdAt"),
        "schools_all": school_names,
        "type_code": detail.get("type"),
        "user": detail.get("user"),
        "prompt": {
            "id": (detail.get("prompt") or {}).get("id"),
            "wordMin": (detail.get("prompt") or {}).get("wordMin"),
            "wordMax": (detail.get("prompt") or {}).get("wordMax"),
        },
        "page_url": f"{SITE}/essay-library/{eid}",
    }

    return {
        "school": school,
        "grade": grade,
        "essay_type": essay_type,
        "essay_text": essay_text,
        "metadata": json.dumps(meta, ensure_ascii=False),
    }


def scrape(
    out_csv: str,
    grades: str = "0,1",
    max_essays: int | None = None,
) -> None:
    log = logging.getLogger(__name__)
    sess = session()

    log.info("Fetching browse API (grades=%s)...", grades)
    summaries = iter_browse_essays(sess, grades, max_summaries=max_essays)
    log.info("Browse returned %s essay summaries", len(summaries))

    fieldnames = ["school", "grade", "essay_type", "essay_text", "metadata"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, summary in enumerate(summaries):
            eid = summary.get("id")
            try:
                if i > 0:
                    polite_sleep()
                row = row_from_essay(sess, summary)
                writer.writerow(row)
                f.flush()
                log.info("[%s/%s] OK essay id=%s", i + 1, len(summaries), eid)
            except Exception as ex:
                log.warning("Skip essay id=%s: %s", eid, ex)
                continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape EssaysThatWorked essay library to CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="essaysthatworked_essays.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--grades",
        default="0,1",
        help='Grade tier filter for the browse API (default "0,1" = A+ and A)',
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N essays after listing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    scrape(args.output, grades=args.grades, max_essays=args.max)


if __name__ == "__main__":
    main()

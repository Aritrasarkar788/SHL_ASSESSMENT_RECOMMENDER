"""
SHL Catalog Scraper — Production Version
==========================================
Written from the REAL page structure of shl.com/products/product-catalog/

KEY FINDINGS from inspecting the live page:
─────────────────────────────────────────────
1. URL pattern: https://www.shl.com/products/product-catalog/
   (NOT /solutions/products/ — catalog lives under /products/)

2. Pagination uses PLAIN query params — NO JavaScript needed:
     ?start=0&type=1   → Individual Test Solutions, page 1
     ?start=12&type=1  → page 2
     ?start=372&type=1 → page 32 (last page — confirmed from live pagination)
   type=1 = Individual Test Solutions  ← we want this
   type=2 = Pre-packaged Job Solutions ← IGNORE

3. Each page shows TWO tables. We target the one with header:
   "Individual Test Solutions"

4. Table columns per row:
   [0] name + <a href="/products/product-catalog/view/slug/">
   [1] Remote Testing  — <img> present = "Yes", else "No"
   [2] Adaptive/IRT    — <img> present = "Yes", else "No"
   [3] Test Type       — space-separated letters: K, A B P, etc.

5. Total: 32 pages × 12 items = up to 384 Individual Test Solutions

6. requests + BeautifulSoup is SUFFICIENT for listing pages.
   No Playwright needed — tables are server-rendered HTML.

Run: python scraper/scrape_catalog.py
Run (fast mode): python scraper/scrape_catalog.py --skip-detail
"""

import json
import os
import re
import time
from collections import Counter

import requests
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────
BASE_URL       = "https://www.shl.com"
CATALOG_URL    = "https://www.shl.com/products/product-catalog/"
ITEMS_PER_PAGE = 12
OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "../data/assessments.json")
VALID_TYPES    = {"A", "B", "C", "D", "E", "K", "P", "S"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Utilities ──────────────────────────────────────────────────────────────

def get_page(url: str, retries: int = 3) -> BeautifulSoup:
    """Fetch a URL with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            print(f"    [WARN] Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def has_checkmark(cell) -> str:
    """Detect if a table cell contains a checkmark (Yes) indicator."""
    if cell.find("img"):        # SHL uses <img> for green ticks
        return "Yes"
    text = cell.get_text(strip=True)
    if text in {"✓", "✔", "Yes", "yes", "y", "1"}:
        return "Yes"
    if cell.find(class_=re.compile(r"check|tick|yes|true|icon", re.I)):
        return "Yes"
    return "No"


def parse_test_types(cell) -> list:
    """
    Extract test type letter codes from the Type column.
    Handles both badge spans and plain text like 'K' or 'A B P'.
    """
    types = []
    # Try individual span/div elements first
    for el in cell.find_all(["span", "div"]):
        t = el.get_text(strip=True).upper()
        if t in VALID_TYPES:
            types.append(t)
    # Fallback: plain text split
    if not types:
        for token in cell.get_text(strip=True).upper().split():
            if token in VALID_TYPES:
                types.append(token)
    return list(dict.fromkeys(types))  # dedup, keep order


def get_total_pages(soup: BeautifulSoup) -> int:
    """
    Parse the last page number from pagination.
    Pagination HTML: <a href="?start=372&type=1">32</a>
    """
    max_start = 0
    for a in soup.find_all("a", href=re.compile(r"start=\d+&type=1")):
        m = re.search(r"start=(\d+)", a["href"])
        if m:
            max_start = max(max_start, int(m.group(1)))
    return (max_start // ITEMS_PER_PAGE) + 1 if max_start else 32


# ── Step 1: Listing Pages ──────────────────────────────────────────────────

def scrape_listing_pages() -> list:
    """
    Scrape all paginated listing pages.
    Returns list of dicts with: name, url, remote_support, adaptive_support, test_type
    """
    print("[1/2] Scraping catalog listing pages...")

    # Page 1: also used to detect total pages
    first_url = f"{CATALOG_URL}?start=0&type=1"
    print(f"  Fetching page 1: {first_url}")
    first_soup = get_page(first_url)
    total_pages = get_total_pages(first_soup)
    print(f"  Total pages: {total_pages}  (~{total_pages * ITEMS_PER_PAGE} assessments)")

    assessments = []
    seen_urls   = set()

    for page_num in range(total_pages):
        start = page_num * ITEMS_PER_PAGE
        if page_num == 0:
            soup = first_soup
        else:
            url = f"{CATALOG_URL}?start={start}&type=1"
            print(f"  Page {page_num+1}/{total_pages} ...")
            soup = get_page(url)
            time.sleep(0.4)   # polite delay

        # ── Find the correct table ──────────────────────────────────────
        # Page has 2 tables; we want "Individual Test Solutions"
        target_table = None
        for table in soup.find_all("table"):
            th = table.find("th")
            if th and "Individual Test Solutions" in th.get_text():
                target_table = table
                break

        if not target_table:
            print(f"  [WARN] Table not found on page {page_num+1} — skipping")
            continue

        # ── Parse rows ─────────────────────────────────────────────────
        tbody = target_table.find("tbody")
        rows  = tbody.find_all("tr") if tbody else target_table.find_all("tr")[1:]

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            link = cells[0].find("a")
            if not link:
                continue

            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not href:
                continue

            # Build absolute URL and normalise trailing slash
            full_url = (BASE_URL + href) if href.startswith("/") else href
            full_url = full_url.rstrip("/") + "/"

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            assessments.append({
                "name"            : name,
                "url"             : full_url,
                "remote_support"  : has_checkmark(cells[1]) if len(cells) > 1 else "No",
                "adaptive_support": has_checkmark(cells[2]) if len(cells) > 2 else "No",
                "test_type"       : parse_test_types(cells[3]) if len(cells) > 3 else [],
                "description"     : "",
                "duration"        : 0,
            })

    print(f"  Collected {len(assessments)} Individual Test Solutions")
    return assessments


# ── Step 2: Detail Pages ───────────────────────────────────────────────────

def scrape_detail(url: str) -> dict:
    """
    Scrape an assessment detail page for description + duration.

    Detail page patterns observed:
    - Meta description: cleanest source for a summary sentence
    - Duration appears as "Approximate Completion Time: X minutes" or
      bare "X minutes" in the body
    """
    try:
        soup = get_page(url)

        # ── Description ────────────────────────────────────────────────
        description = ""

        meta = soup.find("meta", {"name": "description"})
        if meta and meta.get("content", "").strip():
            description = meta["content"].strip()

        if not description:
            for sel in [
                "div.product-catalogue-training-calendar__row--description",
                "div.product-hero__description",
                "div.product-catalogue__description",
                ".product-description p",
                "article p",
                "main p",
            ]:
                el = soup.select_one(sel)
                if el and el.get_text(strip=True):
                    description = el.get_text(strip=True)
                    break

        # ── Duration ────────────────────────────────────────────────────
        duration = 0
        text = soup.get_text(" ", strip=True)

        patterns = [
            r"[Aa]pproximate\s+[Cc]ompletion\s+[Tt]ime[^0-9]*(\d+)\s*min",
            r"[Cc]ompletion\s+[Tt]ime[^0-9]*(\d+)\s*min",
            r"[Tt]akes?\s+(\d+)\s*min",
            r"(\d+)\s*-\s*\d+\s*min",          # "15-20 min" → first number
            r"(\d+)\s*min(?:ute)?s?\b",         # bare "15 minutes"
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                val = int(m.group(1))
                if 3 <= val <= 180:             # sanity-check range
                    duration = val
                    break

        return {"description": description[:500], "duration": duration}

    except Exception as e:
        print(f"    [WARN] Detail scrape failed for {url}: {e}")
        return {"description": "", "duration": 0}


def enrich_with_details(assessments: list) -> list:
    """Visit each detail page to fill in description + duration."""
    total = len(assessments)
    print(f"\n[2/2] Scraping {total} detail pages (~{total * 0.5 / 60:.0f} min)...")

    for i, a in enumerate(assessments):
        if i % 25 == 0:
            print(f"  {i+1}/{total} ...")
        detail = scrape_detail(a["url"])
        a["description"] = detail["description"]
        a["duration"]    = detail["duration"]
        time.sleep(0.3)

    return assessments


# ── Save & Stats ───────────────────────────────────────────────────────────

def save(assessments: list, path: str = OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    # Stats
    n          = len(assessments)
    with_desc  = sum(1 for a in assessments if a["description"])
    with_dur   = sum(1 for a in assessments if a["duration"] > 0)
    remote     = sum(1 for a in assessments if a["remote_support"] == "Yes")
    adaptive   = sum(1 for a in assessments if a["adaptive_support"] == "Yes")
    type_count = Counter(t for a in assessments for t in a["test_type"])

    print(f"\n{'='*50}")
    print(f"Saved {n} assessments → {path}")
    print(f"  With description : {with_desc}/{n}")
    print(f"  With duration    : {with_dur}/{n}")
    print(f"  Remote testing   : {remote}")
    print(f"  Adaptive/IRT     : {adaptive}")
    print(f"  Test type dist.  : {dict(type_count.most_common())}")
    print(f"{'='*50}")


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Scrape SHL assessment catalog")
    ap.add_argument("--skip-detail", action="store_true",
                    help="Skip detail pages (faster but no descriptions/durations)")
    ap.add_argument("--output", default=OUTPUT_PATH, help="Output JSON path")
    args = ap.parse_args()

    print("=" * 50)
    print("SHL Assessment Catalog Scraper")
    print("=" * 50)

    data = scrape_listing_pages()

    if not args.skip_detail:
        data = enrich_with_details(data)
    else:
        print("\n[2/2] Skipping detail pages (--skip-detail)")

    save(data, args.output)

    if len(data) >= 377:
        print(f"\n✅ PASS — {len(data)} assessments (≥377 required)")
    else:
        print(f"\n❌ FAIL — only {len(data)} assessments scraped (need ≥377)")
        print("   Possible causes:")
        print("   1. SHL changed their page structure — inspect manually")
        print("   2. Network rate limiting — add longer delays and retry")

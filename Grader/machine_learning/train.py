"""
130point.com PSA Data Scraper
==============================
Scrapes pre-aggregated sold eBay PSA listing data from 130point.com.
Much faster than iterating PSA certs — 130point already did the pairing.

Usage:
    python scraper_130point.py --grades 1 2 3 4 5 6 7 8 9 10 --per-grade 400
    python scraper_130point.py --grades 9 10 --per-grade 1000 --keyword charizard

Output:
    training_data/
        grade_1/  *.jpg
        grade_2/  *.jpg
        ...
        grade_10/ *.jpg
        metadata.json
"""

import re
import os
import time
import json
import hashlib
import logging
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL     = "https://130point.com/sales/"
OUTPUT_DIR   = Path("training_data")
RATE_LIMIT   = 1.2          # seconds between page requests (be respectful)
IMAGE_MIN_PX = 200
MAX_RETRIES  = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://130point.com/",
}


# ─── FETCH HELPERS ────────────────────────────────────────────────────────────

def fetch_page(session: requests.Session, url: str, params: dict = None) -> BeautifulSoup | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning("Rate limited — sleeping %d s", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning("Fetch failed (attempt %d): %s", attempt + 1, e)
            time.sleep(3)
    return None


# ─── 130POINT SCRAPER ─────────────────────────────────────────────────────────

class Point130Scraper:
    """
    130point.com aggregates eBay SOLD listings for PSA-graded cards.
    Each result row contains: card name, grade, sale price, image, eBay link.

    Search URL format:
      https://130point.com/sales/?searchString=psa+10+pokemon&sortOrder=0

    sortOrder values:
      0 = Most Recent
      1 = Highest Price
      2 = Lowest Price
    """

    def __init__(self):
        self.session = requests.Session()

    def search(
        self,
        grade: int,
        keyword: str = "pokemon",
        max_results: int = 400,
        sort: int = 0,
    ) -> list[dict]:
        """
        Scrape 130point for PSA <grade> cards matching keyword.
        Paginates automatically until max_results is reached.
        """
        query    = f"psa {grade} {keyword}"
        results  = []
        page     = 1

        log.info("130point: searching PSA %d '%s' (want %d)", grade, keyword, max_results)

        while len(results) < max_results:
            params = {
                "searchString": query,
                "sortOrder":    sort,
                "page":         page,
            }

            soup = fetch_page(self.session, BASE_URL, params=params)
            if not soup:
                log.warning("Failed to fetch page %d for PSA %d", page, grade)
                break

            batch = self._parse_results(soup, grade)
            if not batch:
                log.info("No more results at page %d", page)
                break

            results.extend(batch)
            log.info("  PSA %d page %d → %d items (total %d)", grade, page, len(batch), len(results))

            # Check if there's a next page
            if not self._has_next_page(soup):
                break

            page += 1
            time.sleep(RATE_LIMIT)

        return results[:max_results]

    def _parse_results(self, soup: BeautifulSoup, grade: int) -> list[dict]:
        """Parse one page of 130point search results."""
        items = []

        # 130point result rows — adjust selector if site structure changes
        rows = soup.select("div.sale-item, tr.sale-row, .result-row, table tr")

        # Fallback: look for any rows containing price and image data
        if not rows:
            rows = soup.find_all("tr")

        for row in rows:
            try:
                item = self._parse_row(row, grade)
                if item:
                    items.append(item)
            except Exception as e:
                log.debug("Row parse error: %s", e)

        return items

    def _parse_row(self, row, grade: int) -> dict | None:
        """Extract data from a single result row."""
        # Image
        img_tag = row.find("img")
        img_url = None
        if img_tag:
            img_url = img_tag.get("src") or img_tag.get("data-src") or ""
            if img_url and not img_url.startswith("http"):
                img_url = "https://130point.com" + img_url
            # Upgrade eBay thumbnail to full resolution
            img_url = re.sub(r's-l\d+\.jpg', 's-l1600.jpg', img_url)

        # Link to original eBay listing
        links    = row.find_all("a", href=True)
        ebay_url = ""
        for link in links:
            href = link["href"]
            if "ebay.com" in href or "ebay" in href.lower():
                ebay_url = href
                break

        # Title / card name
        title = ""
        for tag in ["td", "div", "span", "p"]:
            el = row.find(tag, class_=re.compile(r"title|name|desc|item", re.I))
            if el and el.get_text(strip=True):
                title = el.get_text(strip=True)
                break
        if not title:
            # Grab all text and take the longest chunk
            texts = [t.strip() for t in row.stripped_strings if len(t.strip()) > 10]
            title = max(texts, key=len) if texts else ""

        # Price
        price    = 0.0
        price_el = row.find(string=re.compile(r'\$[\d,]+\.?\d*'))
        if price_el:
            price_str = re.search(r'[\d,]+\.?\d*', price_el)
            if price_str:
                price = float(price_str.group().replace(",", ""))

        # Verify this is actually a PSA card (title sanity check)
        if not img_url or not title:
            return None
        if not re.search(r'psa|pokemon|pokémon|tcg|holo|charizard|pikachu', title, re.I):
            return None
        # Confirm grade mentioned in title matches requested grade
        if not re.search(rf'\bpsa\s*{grade}\b', title, re.I):
            # Still accept if no grade mentioned — 130point search is grade-specific
            pass

        return {
            "grade":   grade,
            "title":   title,
            "images":  [img_url] if img_url else [],
            "url":     ebay_url,
            "price":   price,
            "source":  "130point",
        }

    def _has_next_page(self, soup: BeautifulSoup) -> bool:
        """Check if a 'next page' link exists."""
        next_link = soup.find("a", string=re.compile(r'next|›|»', re.I))
        if next_link:
            return True
        # Also check for pagination with page numbers
        pagination = soup.find(class_=re.compile(r'paginat|page-nav', re.I))
        if pagination:
            current = pagination.find(class_=re.compile(r'active|current', re.I))
            if current and current.find_next_sibling("a"):
                return True
        return False


# ─── IMAGE DOWNLOADER ─────────────────────────────────────────────────────────

def url_to_filename(url: str, grade: int, index: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"grade_{grade}_{index:05d}_{url_hash}.jpg"


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            return img.size[0] >= IMAGE_MIN_PX and img.size[1] >= IMAGE_MIN_PX
    except Exception:
        return False


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists() and validate_image(dest):
        return True
    try:
        resp = session.get(url, timeout=20, stream=True, headers=HEADERS)
        resp.raise_for_status()
        if "image" not in resp.headers.get("Content-Type", ""):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        if not validate_image(dest):
            dest.unlink(missing_ok=True)
            return False
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def bulk_download(records: list[dict], output_dir: Path):
    session   = requests.Session()
    success   = 0
    failed    = 0
    per_grade = {}

    queue = []
    for record in records:
        grade = record["grade"]
        per_grade.setdefault(grade, 0)
        for img_url in record.get("images", []):
            if not img_url:
                continue
            idx      = per_grade[grade]
            filename = url_to_filename(img_url, grade, idx)
            dest     = output_dir / f"grade_{grade}" / filename
            queue.append((img_url, dest))
            per_grade[grade] += 1

    log.info("Downloading %d images…", len(queue))
    with tqdm(queue, desc="Downloading") as pbar:
        for url, dest in pbar:
            if download_image(session, url, dest):
                success += 1
            else:
                failed += 1
            pbar.set_postfix(ok=success, fail=failed)
            time.sleep(0.15)

    # Summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    total = 0
    for grade in range(1, 11):
        folder = output_dir / f"grade_{grade}"
        count  = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        bar    = "█" * (count // 10)
        status = "✓" if count >= 100 else "⚠ Low" if count > 0 else "✗ Empty"
        print(f"  PSA {grade:2d}: {count:4d} images  {bar}  [{status}]")
        total += count
    print(f"  TOTAL : {total}")
    print("=" * 50)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="130point.com PSA data collector")
    parser.add_argument("--grades",    nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--keyword",   default="pokemon", help="Search keyword")
    parser.add_argument("--per-grade", type=int, default=400, help="Max results per grade")
    parser.add_argument("--output",    default="training_data")
    parser.add_argument("--no-download", action="store_true", help="Collect metadata only, skip image download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    scraper    = Point130Scraper()
    all_records = []

    for grade in args.grades:
        records = scraper.search(
            grade=grade,
            keyword=args.keyword,
            max_results=args.per_grade,
        )
        all_records.extend(records)
        log.info("Grade %d: %d records collected", grade, len(records))
        time.sleep(RATE_LIMIT)

    print(f"\nTotal records: {len(all_records)}")

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata_130point.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"Metadata saved → {meta_path}")

    if not args.no_download:
        bulk_download(all_records, output_dir)


if __name__ == "__main__":
    main()
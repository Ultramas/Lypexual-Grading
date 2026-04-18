"""
eBay Sold Listings Scraper — Playwright (Real Browser)
=======================================================
Uses a real Chromium browser to bypass eBay's bot challenge.
Scrapes sold PSA-graded Pokemon card listings for training data.

Setup:
    pip install playwright beautifulsoup4 pillow tqdm requests
    playwright install chromium

Usage:
    python scraper_playwright.py --grades 1 2 3 4 5 6 7 8 9 10 --per-grade 400
    python scraper_playwright.py --grades 9 10 --per-grade 800 --keyword charizard
    python scraper_playwright.py --grades 9 10 --headed   # show browser window
"""

import re
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
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR  = Path("training_data")
IMAGE_MIN   = 200
RATE_LIMIT  = 1.8    # seconds between page navigations
MAX_PAGES   = 20


# ── SCRAPER ───────────────────────────────────────────────────────────────────

class EbayPlaywrightScraper:

    def __init__(self, headed: bool = False):
        self.headed = headed
        self._pw   = None
        self._browser = None
        self._page    = None

    def start(self):
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=not self.headed,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        # Mask webdriver flag
        ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        """)
        self._page = ctx.new_page()

        # Visit homepage first to get cookies
        log.info("Warming up browser on eBay homepage…")
        self._page.goto("https://www.ebay.com", wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

    def stop(self):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def fetch_page(self, grade: int, keyword: str, page: int) -> BeautifulSoup | None:
        url = (
            f"https://www.ebay.com/sch/i.html"
            f"?_nkw=PSA+{grade}+{keyword.replace(' ', '+')}"
            f"&LH_Sold=1&LH_Complete=1&_sacat=2536&_ipg=240&_pgn={page}"
        )
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for listings to appear
            try:
                self._page.wait_for_selector(
                    "li.s-item, .srp-results .s-item",
                    timeout=10000,
                )
            except PWTimeout:
                # Check if we got a challenge page
                if "Pardon Our Interruption" in self._page.title() or \
                   "Checking your browser" in self._page.content():
                    log.warning("Bot challenge hit — waiting 10s for it to resolve…")
                    time.sleep(10)
                    self._page.wait_for_load_state("networkidle", timeout=20000)
                else:
                    log.warning("No listings selector found on page %d", page)

            html = self._page.content()
            return BeautifulSoup(html, "html.parser")

        except Exception as e:
            log.error("Page load error: %s", e)
            return None

    def parse_listings(self, soup: BeautifulSoup, grade: int) -> list[dict]:
        results = []

        # Try multiple possible container selectors
        items = (
            soup.select("li.s-item") or
            soup.select(".srp-results li[id]") or
            soup.select("[data-viewport]") or
            []
        )

        log.debug("Found %d raw item containers", len(items))

        for item in items:
            try:
                rec = self._parse_item(item, grade)
                if rec:
                    results.append(rec)
            except Exception as e:
                log.debug("Parse error: %s", e)

        return results

    def _parse_item(self, item, grade: int) -> dict | None:
        # ── Title ──────────────────────────────────────────────────────────
        title_el = (
            item.select_one(".s-item__title") or
            item.select_one("h3") or
            item.select_one("[class*='title']")
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True)

        if title.lower().startswith("shop on ebay"):
            return None

        # Confirm grade matches
        if not re.search(rf'\bpsa\s*{grade}\b', title, re.I):
            return None

        # Reject if a different PSA grade appears in the title
        for og in [g for g in range(1, 11) if g != grade]:
            if re.search(rf'\bpsa\s*{og}\b', title, re.I):
                return None

        # ── Image ──────────────────────────────────────────────────────────
        img_el  = item.select_one("img")
        img_url = ""
        if img_el:
            img_url = (
                img_el.get("data-src") or
                img_el.get("src") or
                img_el.get("data-lazy-src") or ""
            )
            # Force max-res eBay image
            img_url = re.sub(r's-l\d+\.(jpg|jpeg|webp)', r's-l1600.\1', img_url)

        if not img_url or img_url.startswith("data:") or "gif" in img_url:
            return None

        # ── URL ────────────────────────────────────────────────────────────
        link_el  = item.select_one("a[href*='ebay.com/itm'], a.s-item__link")
        item_url = ""
        if link_el:
            item_url = link_el.get("href", "").split("?")[0]

        # ── Price ──────────────────────────────────────────────────────────
        price_el  = item.select_one(".s-item__price")
        price_str = price_el.get_text(strip=True) if price_el else ""
        price     = 0.0
        m = re.search(r'[\d,]+\.?\d*', price_str.replace(",", ""))
        if m:
            try:
                price = float(m.group())
            except ValueError:
                pass

        return {
            "grade":   grade,
            "title":   title,
            "images":  [img_url],
            "url":     item_url,
            "price":   price,
            "source":  "ebay_playwright",
        }

    def has_next_page(self, soup: BeautifulSoup) -> bool:
        return bool(
            soup.select_one("a[aria-label*='Next'], a.pagination__next, [class*='pagination'] a[rel='next']")
        )

    def collect_grade(
        self,
        grade:       int,
        keyword:     str = "pokemon card",
        max_results: int = 400,
    ) -> list[dict]:
        all_items = []
        log.info("── PSA %d (target %d) ──────────────", grade, max_results)

        for page in range(1, MAX_PAGES + 1):
            if len(all_items) >= max_results:
                break

            soup = self.fetch_page(grade, keyword, page)
            if not soup:
                break

            batch = self.parse_listings(soup, grade)
            if not batch:
                log.info("  No items found on page %d — done with PSA %d", page, grade)
                break

            all_items.extend(batch)
            log.info("  page %2d: +%3d  (total %d)", page, len(batch), len(all_items))

            if not self.has_next_page(soup):
                break

            time.sleep(RATE_LIMIT)

        log.info("PSA %d: collected %d listings\n", grade, min(len(all_items), max_results))
        return all_items[:max_results]


# ── IMAGE DOWNLOADER ──────────────────────────────────────────────────────────

def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            return img.size[0] >= IMAGE_MIN and img.size[1] >= IMAGE_MIN
    except Exception:
        return False


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists() and validate_image(dest):
        return True
    try:
        resp = session.get(url, timeout=20, stream=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0",
            "Referer": "https://www.ebay.com/",
        })
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
    per_grade = {g: 0 for g in range(1, 11)}
    queue     = []

    for rec in records:
        grade = int(rec["grade"])
        for url in rec.get("images", []):
            if not url:
                continue
            h        = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"grade_{grade}_{per_grade[grade]:05d}_{h}.jpg"
            dest     = output_dir / f"grade_{grade}" / filename
            queue.append((url, dest))
            per_grade[grade] += 1

    log.info("Downloading %d images…", len(queue))
    ok = fail = 0

    with tqdm(queue, desc="Downloading") as pbar:
        for url, dest in pbar:
            ok   += download_image(session, url, dest)
            fail += not download_image(session, url, dest)
            pbar.set_postfix(ok=ok, fail=fail)
            time.sleep(0.1)

    _summary(output_dir)


def _summary(output_dir: Path):
    print("\n" + "=" * 52)
    print("  DATASET SUMMARY")
    print("=" * 52)
    total = 0
    for g in range(1, 11):
        folder = output_dir / f"grade_{g}"
        count  = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        bar    = "█" * (count // 10)
        flag   = "✓" if count >= 100 else "⚠" if count > 0 else "✗"
        print(f"  PSA {g:2d}:  {count:4d} imgs  {bar:<20}  {flag}")
        total += count
    print("-" * 52)
    print(f"  TOTAL :  {total}")
    print("=" * 52)
    print(f"\n  Output: {output_dir.resolve()}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades",      nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--keyword",     default="pokemon card")
    parser.add_argument("--per-grade",   type=int, default=400)
    parser.add_argument("--output",      default="training_data")
    parser.add_argument("--headed",      action="store_true", help="Show browser window")
    parser.add_argument("--no-download", action="store_true", help="Skip image download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = EbayPlaywrightScraper(headed=args.headed)
    all_records = []

    try:
        scraper.start()
        for grade in args.grades:
            records = scraper.collect_grade(
                grade=grade,
                keyword=args.keyword,
                max_results=args.per_grade,
            )
            all_records.extend(records)
            time.sleep(2)
    finally:
        scraper.stop()

    print(f"\nTotal records collected: {len(all_records)}")

    meta = output_dir / "metadata.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"Metadata → {meta}")

    if not args.no_download:
        bulk_download(all_records, output_dir)


if __name__ == "__main__":
    main()
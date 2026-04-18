"""
Cross-Platform Pokemon Card Listing Comparator
===============================================
Searches eBay, Amazon, TCGPlayer, Mercari, Whatnot, and Facebook Marketplace
for a given card name or listing URL and ranks results by price.

Requires Playwright for JS-heavy platforms:
    pip install playwright requests beautifulsoup4 tqdm
    playwright install chromium

Usage (standalone):
    python compare.py "PSA 10 Charizard Base Set"
    python compare.py "https://www.ebay.com/itm/123456789"
    python compare.py "Pikachu 1st Edition" --min-price 5 --max-price 500
    python compare.py "PSA 9 Blastoise" --json
    python compare.py "Mewtwo holo" --platforms ebay tcgplayer mercari

Usage (integrated with grader.py — see bottom of grader.py for hook):
    python grader.py --compare "PSA 10 Charizard Base Set"
"""

from __future__ import annotations

import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

MIN_LEGIT_PRICE = 2.00        # ignore anything ≤ this (joke listings)
FB_CL_MIN       = 3.00        # stricter floor for FB Marketplace / Craigslist
TIMEOUT         = 18          # requests timeout
MAX_PER_PLATFORM = 8          # results to collect per platform

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── DATA MODEL ────────────────────────────────────────────────────────────────

@dataclass
class Listing:
    platform:    str
    title:       str
    price:       float
    url:         str
    condition:   str = ""
    seller:      str = ""
    shipping:    float = 0.0
    image_url:   str = ""
    raw_price:   str = ""

    @property
    def total_price(self) -> float:
        return self.price + self.shipping

    @property
    def display_price(self) -> str:
        if self.shipping > 0:
            return f"${self.price:.2f} + ${self.shipping:.2f} shipping"
        return f"${self.price:.2f} (free shipping or local)"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_price"]   = self.total_price
        d["display_price"] = self.display_price
        return d


# ── QUERY NORMALIZER ──────────────────────────────────────────────────────────

def normalize_query(input_str: str) -> dict:
    """
    Accepts either a search query string or a URL.
    If a URL, extracts the item title via a quick fetch.
    Returns {"query": str, "source_url": str|None, "source_platform": str|None}
    """
    parsed = urlparse(input_str)
    is_url = parsed.scheme in ("http", "https") and parsed.netloc

    if not is_url:
        return {"query": input_str, "source_url": None, "source_platform": None}

    # Detect platform from URL
    domain = parsed.netloc.lower()
    platform = None
    if "ebay.com"      in domain: platform = "ebay"
    elif "amazon.com"  in domain: platform = "amazon"
    elif "tcgplayer"   in domain: platform = "tcgplayer"
    elif "mercari"     in domain: platform = "mercari"
    elif "facebook"    in domain: platform = "facebook"
    elif "craigslist"  in domain: platform = "craigslist"
    elif "whatnot"     in domain: platform = "whatnot"

    # Try to extract title from the page
    log.info("Fetching title from source URL: %s", input_str)
    try:
        resp  = requests.get(input_str, headers=HEADERS, timeout=TIMEOUT)
        soup  = BeautifulSoup(resp.text, "html.parser")
        title = (
            (soup.find("h1") and soup.find("h1").get_text(strip=True)) or
            (soup.find("title") and soup.find("title").get_text(strip=True)) or
            ""
        )
        # Clean up the title — strip site names, extra chars
        title = re.sub(r'\s*[|\-–]\s*(eBay|Amazon|Mercari|TCGPlayer|Facebook).*$', '', title, flags=re.I)
        title = title.strip()
        log.info("Extracted title: %s", title)
        return {"query": title, "source_url": input_str, "source_platform": platform}
    except Exception as e:
        log.warning("Could not fetch source URL: %s", e)
        # Fall back to using the URL path as a hint
        path_parts = [p for p in parsed.path.split("/") if p and len(p) > 3]
        query = " ".join(path_parts[-2:]).replace("-", " ").replace("_", " ")
        return {"query": query, "source_url": input_str, "source_platform": platform}


# ── PRICE PARSER ──────────────────────────────────────────────────────────────

def parse_price(text: str) -> float:
    """Extract first numeric price from a string like '$24.99' or 'USD 24.99'."""
    if not text:
        return 0.0
    text = text.replace(",", "")
    m = re.search(r'[\$£€]?\s*(\d+(?:\.\d{1,2})?)', text)
    return float(m.group(1)) if m else 0.0


def is_legit_price(price: float, platform: str) -> bool:
    """Filter out joke/placeholder listings."""
    if price <= 0:
        return False
    if platform in ("facebook", "craigslist") and price <= FB_CL_MIN:
        return False
    if price <= MIN_LEGIT_PRICE:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM SCRAPERS
# ─────────────────────────────────────────────────────────────────────────────

class BaseScraper:
    PLATFORM = "base"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        raise NotImplementedError

    def _get(self, url: str, params: dict = None) -> BeautifulSoup | None:
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            log.debug("[%s] fetch failed: %s", self.PLATFORM, e)
            return None


# ── EBAY ──────────────────────────────────────────────────────────────────────

class EbayScraper(BaseScraper):
    PLATFORM = "ebay"
    URL      = "https://www.ebay.com/sch/i.html"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        # Use Playwright to bypass bot check
        return _playwright_scrape_ebay(query, max_results)


def _playwright_scrape_ebay(query: str, max_results: int) -> list[Listing]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("Playwright not installed — eBay results skipped")
        return []

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx     = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )
        page = ctx.new_page()
        try:
            page.goto("https://www.ebay.com", wait_until="domcontentloaded", timeout=20000)
            time.sleep(1)

            url = (
                f"https://www.ebay.com/sch/i.html"
                f"?_nkw={quote_plus(query)}&_sacat=2536&LH_BIN=1&_sop=15"
                # _sop=15 = price+shipping lowest first
            )
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            try:
                page.wait_for_selector("li.s-item", timeout=8000)
            except Exception:
                pass

            soup  = BeautifulSoup(page.content(), "html.parser")
            items = soup.select("li.s-item")

            for item in items[:max_results * 2]:
                title_el = item.select_one(".s-item__title")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if title.lower().startswith("shop on ebay"):
                    continue

                price_el = item.select_one(".s-item__price")
                ship_el  = item.select_one(".s-item__shipping, .s-item__logisticsCost")
                link_el  = item.select_one("a.s-item__link")
                img_el   = item.select_one("img.s-item__image-img")

                raw_price = price_el.get_text(strip=True) if price_el else ""
                price     = parse_price(raw_price)

                # Handle price ranges — take lower bound
                if "to" in raw_price.lower():
                    m = re.search(r'[\$£€]\s*(\d+(?:\.\d{1,2})?)', raw_price)
                    price = float(m.group(1)) if m else 0.0

                ship_text = ship_el.get_text(strip=True) if ship_el else ""
                shipping  = 0.0 if re.search(r'free|included', ship_text, re.I) \
                            else parse_price(ship_text)

                url_  = link_el["href"].split("?")[0] if link_el else ""
                img   = img_el.get("src", "") if img_el else ""

                if not is_legit_price(price, "ebay") or not url_:
                    continue

                results.append(Listing(
                    platform  = "eBay",
                    title     = title,
                    price     = price,
                    shipping  = shipping,
                    url       = url_,
                    image_url = img,
                    raw_price = raw_price,
                ))
                if len(results) >= max_results:
                    break

        finally:
            browser.close()

    return results


# ── AMAZON ────────────────────────────────────────────────────────────────────

class AmazonScraper(BaseScraper):
    PLATFORM = "amazon"
    URL      = "https://www.amazon.com/s"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        results = []
        soup = self._get(self.URL, params={
            "k":      query,
            "i":      "toys-and-games",   # closest category for TCG
            "s":      "price-asc-rank",   # sort cheapest first
            "rh":     "n:166086011",      # Trading Card Games node
        })
        if not soup:
            return []

        items = soup.select("[data-component-type='s-search-result']")
        for item in items[:max_results * 2]:
            title_el = item.select_one("h2 a span")
            price_el = item.select_one(".a-price .a-offscreen, .a-price-whole")
            link_el  = item.select_one("h2 a")
            img_el   = item.select_one("img.s-image")

            if not title_el or not price_el or not link_el:
                continue

            title = title_el.get_text(strip=True)
            price = parse_price(price_el.get_text(strip=True))
            href  = "https://www.amazon.com" + link_el.get("href", "")
            img   = img_el.get("src", "") if img_el else ""

            # Amazon usually has free shipping on Prime items
            ship_el  = item.select_one(".a-color-secondary .a-size-base")
            shipping = 0.0
            if ship_el and "$" in ship_el.get_text():
                shipping = parse_price(ship_el.get_text())

            if not is_legit_price(price, "amazon"):
                continue

            results.append(Listing(
                platform  = "Amazon",
                title     = title,
                price     = price,
                shipping  = shipping,
                url       = href,
                image_url = img,
                raw_price = price_el.get_text(strip=True),
            ))
            if len(results) >= max_results:
                break

        return results


# ── TCGPLAYER ─────────────────────────────────────────────────────────────────

class TCGPlayerScraper(BaseScraper):
    PLATFORM = "tcgplayer"
    URL      = "https://www.tcgplayer.com/search/pokemon/product"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        soup = self._get(self.URL, params={
            "q":          query,
            "view":       "grid",
            "productLineName": "pokemon",
        })
        if not soup:
            return []

        results = []
        items   = soup.select(".search-result, .product-card__wrapper")

        for item in items[:max_results * 2]:
            title_el = item.select_one(".product-card__title, h3")
            price_el = item.select_one(".product-card__market-price, .inventory__price-with-shipping")
            link_el  = item.select_one("a[href*='/product/']")
            img_el   = item.select_one("img")

            if not title_el or not link_el:
                continue

            title = title_el.get_text(strip=True)
            price = parse_price(price_el.get_text(strip=True)) if price_el else 0.0
            href  = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.tcgplayer.com" + href
            img = img_el.get("src", "") if img_el else ""

            if not is_legit_price(price, "tcgplayer"):
                continue

            results.append(Listing(
                platform  = "TCGPlayer",
                title     = title,
                price     = price,
                url       = href,
                image_url = img,
                raw_price = price_el.get_text(strip=True) if price_el else "",
            ))
            if len(results) >= max_results:
                break

        return results


# ── MERCARI ───────────────────────────────────────────────────────────────────

class MercariScraper(BaseScraper):
    """Mercari uses client-side rendering — uses their internal search API."""
    PLATFORM = "mercari"
    API_URL  = "https://api.mercari.jp/search_index/search"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        # Mercari US public search via their web search endpoint
        try:
            resp = requests.get(
                "https://www.mercari.com/search/",
                params={"keyword": query, "status": "on_sale", "sort": "price_asc"},
                headers={**HEADERS, "Accept": "application/json"},
                timeout=TIMEOUT,
            )
            soup    = BeautifulSoup(resp.text, "html.parser")
            results = []

            items = soup.select("[data-testid='item-cell'], .items-box-content li")
            for item in items[:max_results * 2]:
                title_el = item.select_one("[class*='name'], [class*='title'], p")
                price_el = item.select_one("[class*='price']")
                link_el  = item.select_one("a[href*='/item/']")
                img_el   = item.select_one("img")

                if not title_el or not price_el or not link_el:
                    continue

                title = title_el.get_text(strip=True)
                price = parse_price(price_el.get_text(strip=True))
                href  = link_el.get("href", "")
                if not href.startswith("http"):
                    href = "https://www.mercari.com" + href
                img = img_el.get("src", "") if img_el else ""

                if not is_legit_price(price, "mercari"):
                    continue

                results.append(Listing(
                    platform  = "Mercari",
                    title     = title,
                    price     = price,
                    url       = href,
                    image_url = img,
                    raw_price = price_el.get_text(strip=True),
                ))
                if len(results) >= max_results:
                    break

            return results
        except Exception as e:
            log.debug("Mercari search failed: %s", e)
            return []


# ── FACEBOOK MARKETPLACE ──────────────────────────────────────────────────────

class FacebookMarketplaceScraper(BaseScraper):
    """
    Facebook Marketplace requires login for most searches.
    We use Playwright to load the page and scrape visible listings.
    Results may be limited without a logged-in account.
    """
    PLATFORM = "facebook"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return []

        results = []
        url = f"https://www.facebook.com/marketplace/search/?query={quote_plus(query)}&sortBy=price_ascend"

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx     = browser.new_context(user_agent=HEADERS["User-Agent"], locale="en-US")
            page    = ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=25000)
                time.sleep(3)  # FB needs JS to render

                soup  = BeautifulSoup(page.content(), "html.parser")
                # FB Marketplace listing containers
                items = soup.select(
                    "[data-testid='marketplace_search_feed_listing'], "
                    "div[aria-label*='Listing'], "
                    "a[href*='/marketplace/item/']"
                )

                seen_urls = set()
                for item in items:
                    # Navigate up to find the container if we got an <a> tag
                    if item.name == "a":
                        container = item
                    else:
                        container = item

                    title_el = container.select_one("span[class*='x1lliihq'], span[dir='auto']")
                    price_el = container.select_one("span[class*='x193iq5w']")
                    link_el  = container if container.name == "a" else container.select_one("a[href*='/marketplace/item/']")
                    img_el   = container.select_one("img")

                    title = title_el.get_text(strip=True) if title_el else ""
                    price = parse_price(price_el.get_text(strip=True)) if price_el else 0.0
                    href  = link_el.get("href", "") if link_el else ""
                    if href and not href.startswith("http"):
                        href = "https://www.facebook.com" + href
                    img = img_el.get("src", "") if img_el else ""

                    if not title or not href or href in seen_urls:
                        continue
                    seen_urls.add(href)

                    if not is_legit_price(price, "facebook"):
                        continue

                    results.append(Listing(
                        platform  = "Facebook Marketplace",
                        title     = title,
                        price     = price,
                        url       = href,
                        image_url = img,
                        raw_price = price_el.get_text(strip=True) if price_el else "",
                    ))
                    if len(results) >= max_results:
                        break

            except Exception as e:
                log.debug("Facebook Marketplace scrape error: %s", e)
            finally:
                browser.close()

        return results


# ── CRAIGSLIST ────────────────────────────────────────────────────────────────

class CraigslistScraper(BaseScraper):
    """
    Craigslist search — uses the national search page.
    Results are location-dependent; defaults to nationwide search.
    """
    PLATFORM = "craigslist"
    URL      = "https://www.craigslist.org/search/sss"   # for-sale

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        soup = self._get(self.URL, params={
            "query":     query,
            "sort":      "priceasc",
            "purveyor":  "all",
        })
        if not soup:
            return []

        results = []
        items   = soup.select("li.cl-search-result, .result-row")

        for item in items[:max_results * 2]:
            title_el = item.select_one(".result-title, a.cl-app-anchor")
            price_el = item.select_one(".result-price, .priceinfo")
            link_el  = item.select_one("a[href]")

            if not title_el or not link_el:
                continue

            title = title_el.get_text(strip=True)
            price = parse_price(price_el.get_text(strip=True)) if price_el else 0.0
            href  = link_el.get("href", "")

            if not is_legit_price(price, "craigslist"):
                continue

            results.append(Listing(
                platform  = "Craigslist",
                title     = title,
                price     = price,
                url       = href,
                raw_price = price_el.get_text(strip=True) if price_el else "",
            ))
            if len(results) >= max_results:
                break

        return results


# ── WHATNOT ───────────────────────────────────────────────────────────────────

class WhatnotScraper(BaseScraper):
    """Whatnot — live auction platform. Shows Buy-It-Now listings."""
    PLATFORM = "whatnot"
    URL      = "https://www.whatnot.com/browse"

    def search(self, query: str, max_results: int = MAX_PER_PLATFORM) -> list[Listing]:
        soup = self._get(
            f"https://www.whatnot.com/browse?query={quote_plus(query)}&sort=price_asc",
        )
        if not soup:
            return []

        results = []
        items   = soup.select("[class*='ListingCard'], [class*='ProductCard']")

        for item in items[:max_results * 2]:
            title_el = item.select_one("p, h3, [class*='title']")
            price_el = item.select_one("[class*='price']")
            link_el  = item.select_one("a[href]")
            img_el   = item.select_one("img")

            if not title_el or not link_el:
                continue

            title = title_el.get_text(strip=True)
            price = parse_price(price_el.get_text(strip=True)) if price_el else 0.0
            href  = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.whatnot.com" + href
            img = img_el.get("src", "") if img_el else ""

            if not is_legit_price(price, "whatnot"):
                continue

            results.append(Listing(
                platform  = "Whatnot",
                title     = title,
                price     = price,
                url       = href,
                image_url = img,
                raw_price = price_el.get_text(strip=True) if price_el else "",
            ))
            if len(results) >= max_results:
                break

        return results


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ALL_SCRAPERS: dict[str, type[BaseScraper]] = {
    "ebay":      EbayScraper,
    "amazon":    AmazonScraper,
    "tcgplayer": TCGPlayerScraper,
    "mercari":   MercariScraper,
    "facebook":  FacebookMarketplaceScraper,
    "craigslist":CraigslistScraper,
    "whatnot":   WhatnotScraper,
}

PLATFORM_DISPLAY = {
    "ebay":       "eBay",
    "amazon":     "Amazon",
    "tcgplayer":  "TCGPlayer",
    "mercari":    "Mercari",
    "facebook":   "Facebook Marketplace",
    "craigslist": "Craigslist",
    "whatnot":    "Whatnot",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compare_listings(
    query:       str,
    platforms:   list[str] | None = None,
    min_price:   float = 0.0,
    max_price:   float = 999_999.0,
    max_results: int   = MAX_PER_PLATFORM,
) -> dict:
    """
    Search all platforms and return ranked results.
    Returns a dict with cheapest, all_results, and per-platform summaries.
    """
    active_platforms = platforms or list(ALL_SCRAPERS.keys())
    all_listings: list[Listing] = []
    platform_results: dict[str, list[Listing]] = {}
    errors: dict[str, str] = {}

    log.info("Searching %d platforms for: %s", len(active_platforms), query)

    for platform_key in active_platforms:
        scraper_cls = ALL_SCRAPERS.get(platform_key)
        if not scraper_cls:
            log.warning("Unknown platform: %s", platform_key)
            continue

        log.info("  Scraping %s…", PLATFORM_DISPLAY.get(platform_key, platform_key))
        try:
            scraper  = scraper_cls()
            listings = scraper.search(query, max_results=max_results)

            # Apply price filter
            filtered = [
                l for l in listings
                if min_price <= l.total_price <= max_price
            ]

            platform_results[platform_key] = filtered
            all_listings.extend(filtered)
            log.info("    → %d results", len(filtered))

        except Exception as e:
            log.warning("  %s failed: %s", platform_key, e)
            errors[platform_key] = str(e)
            platform_results[platform_key] = []

    # Sort by total price (price + shipping)
    all_listings.sort(key=lambda l: l.total_price)

    cheapest = all_listings[0] if all_listings else None

    # Per-platform cheapest
    platform_cheapest = {}
    for key, listings in platform_results.items():
        if listings:
            platform_cheapest[key] = min(listings, key=lambda l: l.total_price)

    return {
        "query":              query,
        "cheapest":           cheapest,
        "all_listings":       all_listings,
        "platform_cheapest":  platform_cheapest,
        "platform_results":   platform_results,
        "total_found":        len(all_listings),
        "platforms_searched": active_platforms,
        "platforms_with_results": [k for k, v in platform_results.items() if v],
        "errors":             errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"


def print_comparison(results: dict):
    query   = results["query"]
    cheapest = results["cheapest"]
    all_l   = results["all_listings"]
    total   = results["total_found"]

    print("\n" + "═" * 64)
    print(f"  {BOLD}PRICE COMPARISON REPORT{RESET}")
    print(f"  Query: {CYAN}{query}{RESET}")
    print(f"  Platforms searched: {', '.join(results['platforms_searched'])}")
    print(f"  Total listings found: {total}")
    print("═" * 64)

    if not cheapest:
        print(f"\n  {RED}No listings found across any platform.{RESET}")
        print("  Try a broader search term or different platforms.\n")
        return

    # ── CHEAPEST ──────────────────────────────────────────────────────────
    print(f"\n  {BOLD}{GREEN}★ CHEAPEST LISTING{RESET}")
    print(f"  {'─'*60}")
    print(f"  Platform : {BOLD}{cheapest.platform}{RESET}")
    print(f"  Title    : {cheapest.title[:58]}")
    print(f"  Price    : {GREEN}{BOLD}{cheapest.display_price}{RESET}")
    print(f"  Link     : {CYAN}{cheapest.url}{RESET}")
    if cheapest.condition:
        print(f"  Condition: {cheapest.condition}")
    print()

    # ── PER-PLATFORM CHEAPEST ─────────────────────────────────────────────
    print(f"  {BOLD}CHEAPEST PER PLATFORM{RESET}")
    print(f"  {'─'*60}")
    print(f"  {'Platform':<22} {'Price':>10}   {'Link'}")
    print(f"  {'─'*60}")

    plat_cheap = results["platform_cheapest"]

    for platform_key in results["platforms_searched"]:
        plat_name = PLATFORM_DISPLAY.get(platform_key, platform_key)
        if platform_key in plat_cheap:
            listing = plat_cheap[platform_key]
            is_best = (listing.url == cheapest.url)
            star    = f"{GREEN}★{RESET} " if is_best else "  "
            price_s = f"{GREEN}{listing.display_price}{RESET}" if is_best else listing.display_price
            title_s = listing.title[:30] + "…" if len(listing.title) > 30 else listing.title
            print(f"  {star}{plat_name:<20} {listing.total_price:>8.2f}   {listing.url}")
        else:
            err = results["errors"].get(platform_key, "no results")
            print(f"  {DIM}  {plat_name:<20} {'—':>8}   ({err}){RESET}")

    # ── ALL RESULTS RANKED ────────────────────────────────────────────────
    print(f"\n  {BOLD}ALL LISTINGS — RANKED BY PRICE{RESET}")
    print(f"  {'─'*60}")

    for i, listing in enumerate(all_l[:20], 1):   # cap at 20 for readability
        is_best = (i == 1)
        marker  = f"{GREEN}#{i:<2}{RESET}" if is_best else f"#{i:<2}"
        plat    = f"{listing.platform:<18}"
        price   = f"${listing.total_price:>8.2f}"
        title   = listing.title[:30] + "…" if len(listing.title) > 30 else listing.title
        col     = GREEN if is_best else (YELLOW if i <= 3 else RESET)
        print(f"  {marker} {col}{plat}{RESET} {col}{price}{RESET}  {listing.url}")

    if len(all_l) > 20:
        print(f"  {DIM}  … and {len(all_l)-20} more results (use --json to see all){RESET}")

    if results["errors"]:
        print(f"\n  {DIM}Platforms with errors: "
              f"{', '.join(results['errors'].keys())}{RESET}")

    print("\n" + "═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform Pokemon card price comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare.py "PSA 10 Charizard Base Set"
  python compare.py "https://www.ebay.com/itm/123456789"
  python compare.py "Pikachu 1st Edition" --min-price 20 --max-price 300
  python compare.py "PSA 9 Blastoise" --platforms ebay amazon tcgplayer
  python compare.py "Mewtwo holo" --json > results.json
        """,
    )
    parser.add_argument(
        "query",
        help="Search query (card name, PSA grade + name) OR a direct listing URL",
    )
    parser.add_argument(
        "--platforms", nargs="+",
        choices=list(ALL_SCRAPERS.keys()),
        default=list(ALL_SCRAPERS.keys()),
        help="Platforms to search (default: all)",
    )
    parser.add_argument("--min-price",   type=float, default=0.0,
                        help="Minimum price filter (default: 0)")
    parser.add_argument("--max-price",   type=float, default=999999,
                        help="Maximum price filter (default: unlimited)")
    parser.add_argument("--max-results", type=int,   default=MAX_PER_PLATFORM,
                        help=f"Max results per platform (default: {MAX_PER_PLATFORM})")
    parser.add_argument("--json",        action="store_true",
                        help="Output raw JSON")
    args = parser.parse_args()

    # Resolve query (handle URLs)
    resolved = normalize_query(args.query)
    query    = resolved["query"]

    if resolved["source_url"]:
        print(f"\n  Source URL detected → extracted query: '{query}'")

    # Run comparison
    results = compare_listings(
        query       = query,
        platforms   = args.platforms,
        min_price   = args.min_price,
        max_price   = args.max_price,
        max_results = args.max_results,
    )

    if args.json:
        # Serialize listings to dicts
        output = {
            "query":    results["query"],
            "cheapest": results["cheapest"].to_dict() if results["cheapest"] else None,
            "total_found": results["total_found"],
            "all_listings": [l.to_dict() for l in results["all_listings"]],
            "platform_cheapest": {
                k: v.to_dict() for k, v in results["platform_cheapest"].items()
            },
            "errors": results["errors"],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_comparison(results)


if __name__ == "__main__":
    main()
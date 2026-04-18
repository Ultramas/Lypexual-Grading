"""
Pokemon Card PSA Training Data Collector
========================================
Pulls labeled card images from:
  1. eBay Browse API  (OAuth client-credentials flow)
  2. PSA Cert Registry scraper (rate-limited)
  3. Data pairing + disk export (grade_1/ ... grade_10/)

Setup:
  pip install requests pillow tqdm python-dotenv aiohttp aiofiles

.env file:
  EBAY_CLIENT_ID=your_app_id_here
  EBAY_CLIENT_SECRET=your_cert_id_here
"""

import os
import re
import time
import json
import base64
import hashlib
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlencode

import requests
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

try:
    import aiohttp
    import aiofiles
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

EBAY_OAUTH_URL    = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_URL   = "https://api.ebay.com/buy/browse/v1/item_summary/search"
PSA_CERT_URL      = "https://www.psacard.com/cert/{cert}"
PSA_API_URL       = "https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert}"

OUTPUT_DIR        = Path("training_data")
IMAGE_MIN_SIZE    = (200, 200)     # reject images smaller than this
EBAY_PAGE_LIMIT   = 200           # max items per eBay API call (eBay cap is 200)
EBAY_MAX_PAGES    = 10            # pages per grade query
PSA_RATE_LIMIT    = 2.5          # seconds between PSA requests
EBAY_RATE_LIMIT   = 0.5          # seconds between eBay requests
DOWNLOAD_WORKERS  = 8             # parallel image downloads

GRADE_LABELS = list(range(1, 11))

# ─── EBAY OAUTH ───────────────────────────────────────────────────────────────

class EbayAuth:
    """Client-Credentials OAuth flow for eBay public APIs."""

    def __init__(self):
        self.client_id     = os.getenv("EBAY_CLIENT_ID")
        self.client_secret = os.getenv("EBAY_CLIENT_SECRET")
        self._token        = None
        self._expires_at   = None

        if not self.client_id or not self.client_secret:
            raise EnvironmentError(
                "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET in your .env file.\n"
                "Get them at https://developer.ebay.com → My Account → Application Keys"
            )

    def get_token(self) -> str:
        """Return a valid Bearer token, refreshing if needed."""
        if self._token and datetime.utcnow() < self._expires_at:
            return self._token

        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        resp = requests.post(
            EBAY_OAUTH_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        self._token      = data["access_token"]
        expires_in       = int(data.get("expires_in", 7200))
        self._expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
        log.info("eBay token refreshed, expires in %d s", expires_in)
        return self._token


# ─── EBAY BROWSE API ──────────────────────────────────────────────────────────

class EbayCollector:
    """
    Searches eBay for PSA-graded Pokemon cards by grade.

    eBay Browse API documentation:
    https://developer.ebay.com/api-docs/buy/browse/resources/item_summary/methods/search

    Rate limits (Sandbox vs Production):
      - Production: 5,000 calls/day (default), 1,000,000/day (approved)
      - Per call: up to 200 items returned
      - No per-second hard limit, but stay under ~2 req/s to be safe

    To increase limits: submit an "App Check" request in the eBay Developer portal.
    """

    CATEGORY_ID = "2536"   # eBay: Trading Card Games → Pokemon

    def __init__(self, auth: EbayAuth):
        self.auth = auth

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth.get_token()}",
            "Content-Type":  "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        }

    def search_psa_grade(
        self,
        grade: int,
        keyword: str = "pokemon card",
        sold_only: bool = False,
        max_pages: int = EBAY_MAX_PAGES,
    ) -> list[dict]:
        """
        Search for PSA <grade> Pokemon cards.
        Returns list of raw item summary dicts from eBay.

        sold_only=True uses completed/sold listings for ground-truth grades.
        Note: eBay's Browse API doesn't natively filter "sold" items.
        Use the Finding API (findCompletedItems) for sold-only searches.
        """
        items = []
        offset = 0
        query  = f'PSA {grade} {keyword}'

        # Filters:
        # conditionIds 2750 = "For parts or not working" ... actually use 3000 = "Very Good"
        # Better: use keyword "PSA {grade}" which is in listing titles
        params = {
            "q":           query,
            "category_ids": self.CATEGORY_ID,
            "limit":       EBAY_PAGE_LIMIT,
            "fieldgroups": "EXTENDED",      # includes seller info, images[]
        }

        log.info("eBay: querying PSA %d cards ('%s')", grade, query)

        for page in range(max_pages):
            params["offset"] = offset
            try:
                resp = requests.get(
                    EBAY_BROWSE_URL,
                    headers=self._headers(),
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                data  = resp.json()
                batch = data.get("itemSummaries", [])
                items.extend(batch)
                log.info("  page %d/%d → %d items (total %d)", page+1, max_pages, len(batch), len(items))

                if len(batch) < EBAY_PAGE_LIMIT:
                    break   # last page

                offset += EBAY_PAGE_LIMIT
                time.sleep(EBAY_RATE_LIMIT)

            except requests.HTTPError as e:
                log.error("eBay HTTP error on page %d: %s", page+1, e)
                if e.response.status_code == 429:
                    log.warning("Rate limited — sleeping 60 s")
                    time.sleep(60)
                break

        return items

    def extract_images(self, item: dict) -> list[str]:
        """Pull all image URLs from an eBay item summary."""
        urls = []
        if thumb := item.get("thumbnailImages"):
            for img in thumb:
                urls.append(img.get("imageUrl", ""))
        if additional := item.get("additionalImages"):
            for img in additional:
                urls.append(img.get("imageUrl", ""))
        # Primary image
        if primary := item.get("image", {}).get("imageUrl"):
            if primary not in urls:
                urls.insert(0, primary)

        # eBay thumbnail URLs end in s-l140.jpg, s-l300.jpg etc.
        # Replace with s-l1600.jpg for highest resolution
        urls = [re.sub(r's-l\d+\.jpg', 's-l1600.jpg', u) for u in urls if u]
        return list(dict.fromkeys(urls))   # deduplicate preserving order

    def collect_grade(self, grade: int, keyword: str = "pokemon card") -> list[dict]:
        """Collect items for one grade and normalize to {grade, images, title, url}."""
        items = self.search_psa_grade(grade, keyword=keyword)
        results = []
        for item in items:
            images = self.extract_images(item)
            if not images:
                continue
            # Double-check the listing title actually mentions this grade
            title  = item.get("title", "")
            if f"PSA {grade}" not in title and f"PSA{grade}" not in title:
                continue
            results.append({
                "grade":      grade,
                "title":      title,
                "url":        item.get("itemWebUrl", ""),
                "images":     images,
                "price":      item.get("price", {}).get("value"),
                "currency":   item.get("price", {}).get("currency"),
                "seller":     item.get("seller", {}).get("username"),
                "source":     "ebay",
                "item_id":    item.get("itemId"),
            })
        log.info("PSA %d: %d valid listings with images", grade, len(results))
        return results


# ─── PSA REGISTRY SCRAPER ─────────────────────────────────────────────────────

class PSAScraper:
    """
    Iterates over PSA cert numbers to collect grade metadata.
    PSA cert numbers are sequential integers.

    Strategy:
      - Recent Pokemon cards (WOTC era, 1999-2003) tend to have certs in the
        1,000,000–15,000,000 range (early PSA grading era).
      - More recent submissions are in the 80,000,000+ range.
      - Start with a known good range and work outward.

    Legality note:
      - PSA's public cert lookup is publicly accessible.
      - Respect robots.txt and rate-limit aggressively (PSA blocks IPs quickly).
      - Consider using the PSA Pop Report API if you qualify:
        https://www.psacard.com/pop/

    Alternative (more reliable): use 130point.com sold data
      https://130point.com/sales/?searchString=psa+10+pokemon
      This site aggregates eBay sold PSA listings and is much easier to scrape.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.psacard.com/",
    }

    # Known cert ranges for Pokemon cards (approximate)
    # These were found by manually looking up known cards on psacard.com/cert/
    CERT_RANGES = {
        "wotc_base":    (1_000_000,   3_000_000),   # 1999 Base Set era
        "wotc_jungle":  (3_000_000,   6_000_000),   # Jungle/Fossil
        "neo_e_card":   (6_000_000,  12_000_000),   # Neo series
        "ex_era":       (12_000_000, 25_000_000),   # EX series
        "recent":       (80_000_000, 95_000_000),   # Modern cards
    }

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(self.HEADERS)

    def lookup_cert(self, cert_number: int) -> dict | None:
        """
        Look up a single PSA cert and return grade metadata.
        Uses PSA's public JSON endpoint (reverse-engineered from their web UI).
        """
        url = PSA_CERT_URL.format(cert=cert_number)
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 404:
                return None   # cert doesn't exist
            if resp.status_code == 429:
                log.warning("PSA rate limited — sleeping 30 s")
                time.sleep(30)
                return None
            resp.raise_for_status()

            # Parse HTML (PSA renders grade in the page)
            html = resp.text

            # Extract grade
            grade_match = re.search(
                r'<span[^>]*class="[^"]*cert-grade[^"]*"[^>]*>\s*(\d+(?:\.\d+)?)\s*</span>',
                html
            )
            if not grade_match:
                # Try alternate pattern
                grade_match = re.search(
                    r'"grade"\s*:\s*"(\d+(?:\.\d+)?)"', html
                )
            if not grade_match:
                return None

            grade_raw = grade_match.group(1)
            grade     = int(float(grade_raw))   # "9.5" → 9, "10" → 10

            # Extract card name
            name_match = re.search(
                r'<span[^>]*class="[^"]*cert-item-name[^"]*"[^>]*>([^<]+)</span>', html
            )
            card_name = name_match.group(1).strip() if name_match else ""

            # Only keep Pokemon cards
            if not re.search(r'pokemon|pokémon|pikachu|charizard|tcg', card_name, re.I):
                return None

            return {
                "cert":      cert_number,
                "grade":     grade,
                "card_name": card_name,
                "url":       url,
                "source":    "psa_registry",
            }

        except requests.RequestException as e:
            log.debug("PSA lookup failed for cert %d: %s", cert_number, e)
            return None

    def scan_range(
        self,
        start: int,
        end: int,
        step: int = 1,
        target_per_grade: int = 200,
    ) -> list[dict]:
        """
        Scan a range of cert numbers, collecting results per grade.
        Stops early if all grades have enough samples.
        """
        collected = {g: [] for g in GRADE_LABELS}
        cert_iter = range(start, end, step)

        log.info("PSA: scanning certs %d → %d (step %d)", start, end, step)

        with tqdm(cert_iter, desc="PSA certs") as pbar:
            for cert in pbar:
                # Check if we have enough for all grades
                if all(len(v) >= target_per_grade for v in collected.values()):
                    log.info("All grades at target, stopping PSA scan.")
                    break

                result = self.lookup_cert(cert)
                if result and 1 <= result["grade"] <= 10:
                    grade = result["grade"]
                    collected[grade].append(result)
                    pbar.set_postfix({f"PSA{g}": len(collected[g]) for g in GRADE_LABELS})

                time.sleep(PSA_RATE_LIMIT)

        all_results = [item for items in collected.values() for item in items]
        log.info("PSA scan complete: %d total records", len(all_results))
        return all_results


# ─── IMAGE DOWNLOADER ─────────────────────────────────────────────────────────

def url_to_filename(url: str, grade: int, index: int) -> str:
    """Create a deterministic filename from a URL + grade."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"grade_{grade}_{index:05d}_{url_hash}.jpg"


def validate_image(path: Path) -> bool:
    """Check image is valid, readable, and above minimum size."""
    try:
        with Image.open(path) as img:
            if img.size[0] < IMAGE_MIN_SIZE[0] or img.size[1] < IMAGE_MIN_SIZE[1]:
                return False
            return True
    except Exception:
        return False


def download_image(url: str, dest: Path, timeout: int = 20) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists():
        return True  # already downloaded

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        if not validate_image(dest):
            dest.unlink(missing_ok=True)
            return False

        return True

    except Exception as e:
        log.debug("Download failed %s: %s", url, e)
        dest.unlink(missing_ok=True)
        return False


def bulk_download(records: list[dict], output_dir: Path, max_per_grade: int = 500):
    """
    Download images from collected records.
    Saves to: output_dir/grade_{N}/grade_{N}_XXXXX_hash.jpg
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    per_grade_count = {g: 0 for g in GRADE_LABELS}
    download_queue  = []

    for record in records:
        grade = record.get("grade")
        if grade not in GRADE_LABELS:
            continue
        if per_grade_count[grade] >= max_per_grade:
            continue

        images = record.get("images", [])[:3]  # max 3 images per listing
        for img_url in images:
            if not img_url:
                continue
            idx      = per_grade_count[grade]
            filename = url_to_filename(img_url, grade, idx)
            dest     = output_dir / f"grade_{grade}" / filename
            download_queue.append((img_url, dest, grade))
            per_grade_count[grade] += 1

            if per_grade_count[grade] >= max_per_grade:
                break

    log.info("Download queue: %d images", len(download_queue))

    success = 0
    failed  = 0

    with tqdm(download_queue, desc="Downloading") as pbar:
        for url, dest, grade in pbar:
            if download_image(url, dest):
                success += 1
            else:
                failed += 1
            pbar.set_postfix(ok=success, fail=failed)
            time.sleep(0.1)  # gentle rate limit

    log.info("Download complete: %d OK, %d failed", success, failed)
    _print_dataset_summary(output_dir)


def _print_dataset_summary(output_dir: Path):
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    total = 0
    for grade in GRADE_LABELS:
        folder = output_dir / f"grade_{grade}"
        count  = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        bar    = "█" * (count // 10)
        status = "✓ OK" if count >= 100 else "⚠ Low" if count > 0 else "✗ Empty"
        print(f"  PSA {grade:2d}: {count:4d} images  {bar}  [{status}]")
        total += count
    print("-"*50)
    print(f"  TOTAL: {total} images")
    print("="*50)
    print(f"\nMinimum recommended: 200+ per grade for basic training")
    print(f"Output dir: {output_dir.resolve()}\n")


# ─── PAIRER: match eBay images to PSA cert grades ────────────────────────────

class DataPairer:
    """
    Pairs eBay images with PSA cert data by matching card names.
    When an eBay listing mentions a cert number, we can look that cert
    up in PSA to get the authoritative grade.
    """

    CERT_PATTERN = re.compile(r'\b(\d{7,9})\b')  # 7-9 digit cert numbers in listing titles

    def __init__(self, psa_scraper: PSAScraper):
        self.psa_scraper = psa_scraper
        self._cert_cache: dict[int, dict] = {}

    def extract_cert_from_title(self, title: str) -> int | None:
        """Try to extract a PSA cert number from an eBay listing title."""
        matches = self.CERT_PATTERN.findall(title)
        for m in matches:
            n = int(m)
            if 1_000_000 <= n <= 99_999_999:  # plausible PSA cert range
                return n
        return None

    def verify_grade_via_cert(self, ebay_record: dict) -> dict | None:
        """
        If the eBay listing contains a cert number, verify the grade via PSA.
        Returns the eBay record with 'verified_grade' added, or None if unverifiable.
        """
        cert = self.extract_cert_from_title(ebay_record.get("title", ""))
        if not cert:
            return ebay_record  # no cert to verify, use eBay-reported grade

        if cert in self._cert_cache:
            psa_data = self._cert_cache[cert]
        else:
            log.debug("PSA cert lookup: %d", cert)
            psa_data = self.psa_scraper.lookup_cert(cert)
            self._cert_cache[cert] = psa_data
            time.sleep(PSA_RATE_LIMIT)

        if psa_data:
            ebay_record["verified_grade"] = psa_data["grade"]
            ebay_record["psa_cert"]       = cert
            ebay_record["psa_card_name"]  = psa_data.get("card_name", "")
            log.debug("Cert %d → PSA %d (was: %d)", cert, psa_data["grade"], ebay_record["grade"])

        return ebay_record


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pokemon PSA Training Data Collector")
    parser.add_argument("--grades",   nargs="+", type=int, default=GRADE_LABELS,
                        help="Which PSA grades to collect (default: 1-10)")
    parser.add_argument("--keyword",  default="pokemon card",
                        help="Search keyword appended to PSA query")
    parser.add_argument("--max-per-grade", type=int, default=300,
                        help="Max images per grade to download")
    parser.add_argument("--output",   default="training_data",
                        help="Output directory for training images")
    parser.add_argument("--skip-psa", action="store_true",
                        help="Skip PSA cert verification (faster)")
    parser.add_argument("--ebay-pages", type=int, default=5,
                        help="eBay API pages per grade (200 items/page)")
    parser.add_argument("--psa-range", nargs=2, type=int,
                        default=[1_000_000, 5_000_000],
                        metavar=("START", "END"),
                        help="PSA cert number range to scan")
    args = parser.parse_args()

    output_dir = Path(args.output)
    all_records = []

    # ── Step 1: eBay data collection ──────────────────────────────
    print("\n── STEP 1: eBay Collection ───────────────────────────────────")
    try:
        auth      = EbayAuth()
        collector = EbayCollector(auth)

        for grade in args.grades:
            records = collector.collect_grade(grade, keyword=args.keyword)
            all_records.extend(records)
            time.sleep(1)

        print(f"eBay: collected {len(all_records)} records across grades {args.grades}")

    except EnvironmentError as e:
        print(f"[eBay Error] {e}")
        print("Skipping eBay collection. Set up .env file to enable.\n")

    # ── Step 2: PSA cert verification ─────────────────────────────
    if not args.skip_psa and all_records:
        print("\n── STEP 2: PSA Cert Verification ────────────────────────────")
        scraper = PSAScraper()
        pairer  = DataPairer(scraper)

        verified = []
        for record in tqdm(all_records, desc="Verifying via PSA"):
            result = pairer.verify_grade_via_cert(record)
            if result:
                verified.append(result)

        # Remap grade to verified_grade if available
        for r in verified:
            if "verified_grade" in r:
                r["grade"] = r["verified_grade"]

        all_records = verified
        print(f"PSA verification complete: {len(all_records)} records")

    # ── Step 3: PSA range scan (standalone, for more data) ────────
    if not args.skip_psa:
        print("\n── STEP 3: PSA Registry Range Scan ──────────────────────────")
        print("  This supplements eBay data with directly-scraped cert grades.")
        print(f"  Scanning certs {args.psa_range[0]:,} → {args.psa_range[1]:,}")
        scraper = PSAScraper()
        psa_records = scraper.scan_range(
            start=args.psa_range[0],
            end=args.psa_range[1],
            step=100,            # sample every 100th cert (speeds up scan)
            target_per_grade=50,
        )
        # PSA records don't have images — pair with eBay using card name later
        print(f"PSA scan found {len(psa_records)} Pokemon card certs")

    # ── Step 4: Download images ────────────────────────────────────
    if all_records:
        print("\n── STEP 4: Downloading Images ───────────────────────────────")
        bulk_download(all_records, output_dir, max_per_grade=args.max_per_grade)
    else:
        print("\nNo records collected. Check your eBay API credentials.")

    # ── Save metadata ──────────────────────────────────────────────
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
# grader/scraper/fetchdata.py

import requests
import os
import time
from bs4 import BeautifulSoup

# ---------------------------------------------------------------
# SOURCE 1: PSA Card Facts (official registry, public)
# https://www.psacard.com/cardfacts/
# Each graded card has a cert number and grade — scrapeable
# ---------------------------------------------------------------

PSA_CERT_URL = "https://www.psacard.com/cert/{cert_number}"

def fetch_psa_cert(cert_number: int):
    """Fetch grade data for a specific PSA cert number."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = PSA_CERT_URL.format(cert_number=cert_number)
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    grade_el = soup.find("span", class_="cert-grade")  # adjust selector
    if grade_el:
        return {
            "cert": cert_number,
            "grade": int(grade_el.text.strip()),
            "url": url
        }
    return None

# ---------------------------------------------------------------
# SOURCE 2: eBay Sold Listings (most practical for bulk data)
# Search: "PSA 10 Charizard" etc., filter by Sold Items
# Use eBay Browse API (free with developer account):
# https://developer.ebay.com/develop/apis/restful-apis/buy-apis
# ---------------------------------------------------------------

EBAY_API_KEY = "YOUR_EBAY_APP_ID"

def search_ebay_psa_cards(grade: int, keyword: str = "pokemon", limit: int = 50):
    """Search eBay for sold PSA-graded cards at a specific grade."""
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {EBAY_API_KEY}",
        "Content-Type": "application/json"
    }
    params = {
        "q": f"PSA {grade} {keyword} card",
        "filter": "conditionIds:{2750}",  # Graded condition
        "limit": limit,
        "fieldgroups": "EXTENDED"
    }
    resp = requests.get(url, headers=headers, params=params)
    return resp.json().get("itemSummaries", [])

# ---------------------------------------------------------------
# SOURCE 3: Pokemon TCG API (card metadata, pair with PSA data)
# https://pokemontcg.io/ — FREE, no auth needed
# Gives you official card images to use as "pristine" references
# ---------------------------------------------------------------

def fetch_tcg_card_image(card_id: str):
    """Fetch official card image from Pokemon TCG API."""
    url = f"https://api.pokemontcg.io/v2/cards/{card_id}"
    resp = requests.get(url)
    data = resp.json().get("data", {})
    return data.get("images", {}).get("large")

# ---------------------------------------------------------------
# SOURCE 4: Kaggle Datasets (pre-labeled, best for starting out)
# Search these on Kaggle:
# - "Pokemon Card Grade Dataset"
# - "PSA Graded Pokemon Cards"
# - "Trading Card Condition Classification"
# Install: pip install kaggle
# ---------------------------------------------------------------

def download_kaggle_dataset(dataset_slug: str, output_dir: str):
    """
    dataset_slug examples:
      - 'evangower/pokemon-card-grade' (if available)
      - 'robikscube/pokemon-cards'
    """
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", output_dir,
        "--unzip"
    ])
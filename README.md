# Pokemon Card PSA Grader

ML-powered grading system for Pokemon trading cards on a 1-10 PSA scale. Uses computer vision + deep learning, plus cross-platform price comparison.

## Quick Start

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Train Model

```bash
python Grader/machine_learning/train.py training_data/
```

Expected folder structure:
```
training_data/
  grade_1/   *.jpg
  grade_2/  *.jpg
  ...
  grade_10/ *.jpg
```

Model outputs to: `Grader/machine_learning/checkpoints/psa_grader.keras`

## Grade a Card

```bash
python Grader/machine_learning/grader.py path/to/card.jpg
python Grader/machine_learning/grader.py --test-dataset training_data/
```

Output includes 4 criteria (each 1-10 with .5 increments):
- **Centering** - border width ratio
- **Corners** - laplacian variance
- **Edges** - perimeter roughness
- **Surface** - scratch detection + luminance

## Compare Prices (Cross-Platform)

```bash
python Grader/machine_learning/compare.py "PSA 10 Charizard Base Set"
python Grader/machine_learning/compare.py "https://www.ebay.com/itm/123456789"
python Grader/machine_learning/compare.py "PSA 9 Blastoise" --platforms ebay amazon tcgplayer mercari
```

Searches: eBay, Amazon, TCGPlayer, Mercari, Facebook Marketplace, Craigslist, Whatnot

## Data Collection (Scrapers)

```bash
# Requires Playwright for eBay (browsers bot detection)
pip install playwright beautifulsoup4 pillow tqdm
playwright install chromium
python Grader/machine_learning/scraper_playwright.py --grades 1 2 3 4 5 6 7 8 9 10 --per-grade 400
```

## Tech Stack

- Django 6 + DRF
- TensorFlow / Keras (EfficientNetB3)
- OpenCV, PIL, NumPy
- Playwright (eBay scraping)
- SQLite (default), PostgreSQL supported

## PSA Grading Scale

| Grade | Name | Key Characteristics |
|-------|------|-------------------|
| 10 | Gem Mint | Perfect centering (55/45), no scratches, sharp corners |
| 9 | Mint | Near-perfect, slight centering allowed |
| 8 | NM-MT | Minor wear on corners, slight scratches |
| 7 | Near Mint | Small corner wear, light scratches |
| 6 | EX-MT | Moderate corner wear, surface scratches |
| 5 | Excellent | Noticeable corner/edge wear |
| 4 | VG-EX | Heavy wear, possible creases |
| 3 | Very Good | Major creases, heavy surface issues |
| 2 | Good | Major defects, staining |
| 1 | Poor | Severe damage (holes, tears, missing pieces) |
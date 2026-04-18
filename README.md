# Marketplace Price Scanner

Browser extension that compares prices across eBay and Amazon when viewing listings on Facebook Marketplace, Nextdoor, or Craigslist.

## How It Works

1. Browse Facebook Marketplace / Nextdoor / Craigslist
2. Click the extension icon to scan the product
3. See eBay + Amazon prices (tax included)
   - eBay: price includes shipping
   - Amazon: toggle Prime to see Prime-eligible pricing
4. Click affiliate link to buy (earns commission)

## Development

**Chrome:**
```bash
# 1. Go to chrome://extensions
# 2. Enable Developer mode
# 3. Click "Load unpacked"
# 4. Select this folder
```

**Firefox:**
```bash
# 1. Go to about:debugging#/runtime/this-firefox
# 2. Click "Load Temporary Add-on..."
# 3. Select manifest.json
```

## Files

- `manifest.json` - Extension config (works on Chrome & Firefox)
- `popup.html/js` - Extension popup UI
- `content.js` - Injected script (scans page)
- `background.js` - Service worker

## Tech Stack

- Vanilla JavaScript
- Chrome Extension Manifest V3 + Firefox Web Extension
- eBay Browse API
- Amazon Product Advertising API

---

# Pokemon Card PSA Grader (Legacy)

ML-powered grading system for Pokemon trading cards. Grades cards on a 1-10 PSA scale using computer vision + deep learning.

## Quick Start

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Train Model

```bash
python Grader/machine_learning/train.py Grader/machine_learning/training_data/
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

## Tech Stack

- Django 6 + DRF
- TensorFlow / Keras (EfficientNetB3)
- OpenCV, PIL, NumPy
- SQLite (default), PostgreSQL supported
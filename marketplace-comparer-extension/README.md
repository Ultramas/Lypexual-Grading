# Marketplace Comparison Extension

Firefox-first browser extension skeleton for comparing listings across eBay, Facebook Marketplace, and Craigslist from a required `title + description` query.

## What This Skeleton Covers
- Manifest V3 extension structure
- Popup UI bones for manual, button-activated controls
- Options page bones for default filters
- First-run consent prompt for cookies/history logging
- Persistent user inputs and preferences across sessions
- Search/view history skeleton gated behind user consent
- Coupon opt-in skeleton
- Background worker skeleton for cross-platform search handoff
- Content script skeleton for reading listing pages and search pages
- Shared normalization/filter placeholders
- Implementation plan in `docs/IMPLEMENTATION_PLAN.md`

## Core Product Goal
- Start from a listing page or a manual form.
- Require both title and description.
- Build a normalized query from those fields.
- Push that query into the search UI of the other supported marketplaces.
- Collect listing signals with user-activated/manual controls rather than passive bulk scraping.
- Normalize price, shipping, taxes, condition, seller standing, seller rating, and distance.
- Let the user refine/filter results with popup buttons and toggles.
- Save form inputs, filters, and preferences across sessions.
- Ask for cookie/history consent on first use before logging searches or viewed listings.
- Optionally search for accessible coupons if the user opts in.

## Supported Rules To Implement
- Distance filters: same city, same state, same country, or any.
- Seller standing filters.
- Optional brand match.
- Seller rating thresholds:
  Default minimum positive ratings: `5`
  Default maximum negatives: `floor(positive / 5)` with a minimum cap of `1`.
- Free shipping filter.
- If the description contains a money amount, prefer that over the visible listing price.
- Flag if `best offer` or `offer` appears in title or description.
- On Facebook Marketplace, ignore and flag `free`, `$1`, or `1$` placeholder prices.
- Taxes use a default rate, but should prefer state-based handling when the user provides a state.
- Seller standing is a ranking boost, not a strict pass/fail requirement.

## Current Project Layout
```text
marketplace-comparer-extension/
  docs/
    ARCHITECTURE.md
    IMPLEMENTATION_PLAN.md
  src/
    adapters/
      craigslist.js
      ebay.js
      facebook.js
    lib/
      coupons.js
      filters.js
      history.js
      mock-data.js
      normalize.js
      ranking.js
      state.js
    background.js
    content.js
    options.html
    options.js
    popup.html
    popup.js
  manifest.json
```

## How The Final Extension Should Work
1. User opens a supported marketplace listing.
2. On first use, the extension asks whether cookies/history logging are allowed.
3. User clicks the extension and presses `Capture Current Listing`.
4. The content script extracts title, description, price, shipping, seller info, condition, and location.
5. The extension refuses to continue unless both title and description are present.
6. The background worker builds a normalized query.
7. The extension opens or updates search pages on the other platforms using that query.
8. The user presses platform-specific `Collect Results` controls when ready.
9. The extension normalizes all captured results and applies filters.
10. The ranking layer computes landed cost and soft boosts.
11. The popup shows flagged items, adjusted total cost, history-aware actions, and filter controls.

## Manual-Control Bias
This skeleton assumes user-triggered collection rather than unattended crawling:
- user clicks buttons to capture the current page
- user clicks buttons to trigger result collection
- user can refine brand/distance/rating filters before final ranking
- user explicitly opts in before history/cookie-backed behavior is enabled

That approach is safer for extension UX and easier to reason about than hidden background scraping.

## Ranking Inputs
- effective price
- taxes
- shipping
- total landed cost
- distance match
- brand match
- seller standing
- seller rating quality
- condition quality
- placeholder-price flags
- offer/best-offer flags
- coupon availability if enabled later

## Notes
- This skeleton does not yet implement live scraping logic.
- This scaffold now stores popup inputs, preferences, and consent state in extension storage.
- Facebook Marketplace and Craigslist DOM selectors will need careful maintenance because markup changes often.
- Tax calculation should use two modes: parsed from page when available, otherwise estimated from a default rate or a state-based rate when the user provides a state.
- Coupon lookup is optional and should remain opt-in.

## Next Build Steps
1. Implement site-specific extractors in `src/content.js`.
2. Implement query normalization in `src/lib/normalize.js`.
3. Implement filtering/ranking in `src/lib/filters.js` and `src/lib/ranking.js`.
4. Add Firefox-targeted testing and manifest validation.
5. Wire `src/content.js` through the adapter layer so the runtime matches the documented structure.
6. Apply options defaults to ranking/filter execution, not just popup restoration.
7. Add real popup rendering for ranked results instead of shell cards.
8. Add optional coupon-provider adapters.

## Chosen Defaults
- Prioritize Firefox first.
- Taxes use a default rate, with better handling when user state is known.
- Seller standing acts as a ranking boost.
- Search/view history only logs after consent.
- Coupon lookup is opt-in.

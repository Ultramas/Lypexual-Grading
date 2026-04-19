# Architecture Notes

## Goal
Keep scraping concerns, normalization, ranking, persistence, and UI separate so real marketplace logic can be added incrementally.

## Intended Layers
- `src/content.js`: browser/page boundary for capture and result collection
- `src/adapters/*.js`: per-platform extraction + search-url behavior
- `src/lib/normalize.js`: query shaping, price hints, offer detection
- `src/lib/filters.js`: hard-rule filtering and cost math
- `src/lib/ranking.js`: ranking-oriented transformations and boost logic
- `src/lib/history.js`: consent-gated history entry formatting
- `src/lib/coupons.js`: optional coupon-provider planning and later adapters
- `src/popup.*`: operator-facing manual controls and status views
- `src/options.*`: durable defaults and tax/rating preferences

## Why This Split
- Marketplace selectors will churn; adapter files isolate that volatility.
- Ranking rules and hard filters will evolve independently.
- Coupon search is optional and should remain detachable.
- History/consent behavior should stay isolated from scraping logic.

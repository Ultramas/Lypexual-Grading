# Implementation Plan

## Scope
Build a Firefox-first browser extension that compares listing opportunities across eBay, Facebook Marketplace, and Craigslist using a required `title + description` input and user-triggered/manual controls.

## Phase 1: Basic Extension Shell
- Add Manifest V3 config.
- Add popup with buttons for:
  `Capture Current Listing`
  `Search Other Platforms`
  `Collect Current Page Results`
  `Apply Filters`
- Add options page for defaults.
- Add first-run consent prompt for cookies/history logging.
- Add storage model for query, filters, collected results, consent, and history.
- Persist user inputs and preferences across sessions.

## Phase 2: Capture Source Listing
- Detect current platform from hostname.
- Extract:
  title
  description
  listing price
  shipping
  condition
  seller name
  seller rating / standing
  location
- Require both title and description before continuing.
- If description contains a currency amount, store it as `descriptionPriceHint` and prefer it over listing price.
- Flag `best offer` / `offer` if found in title or description.
- If consent is granted, add the viewed listing to local history.

## Phase 3: Normalize Query
- Strip marketplace noise words.
- Preserve strong product identifiers:
  brand
  model
  size
  color
  storage / capacity
  edition
- Build two query forms:
  broad query for search bars
  strict token set for post-search filtering
- Persist the latest query state so popup inputs survive browser restarts.

## Phase 4: Push Search To Other Platforms
- Open search URLs or inject query text into search bars on the other marketplaces.
- Use a per-platform adapter:
  `ebay`
  `facebook`
  `craigslist`
- Keep search handoff deterministic and button-activated.
- If consent is granted, log the search event.

## Phase 5: Collect Search Results
- User opens a search results page and clicks `Collect Current Page Results`.
- Extract per result:
  title
  url
  listed price
  description snippet if available
  shipping
  taxes if available
  condition
  seller name
  seller rating / standing
  location
  distance
  best-offer marker
- Mark unsupported/missing fields explicitly instead of guessing.
- Coupon lookup should remain separate and opt-in.

## Phase 6: Apply Business Rules
- Compute `effectivePrice`:
  description price if present, else visible listing price
- Compute `totalCost`:
  effective price + shipping + taxes
- Tax calculation:
  parsed tax if page exposes it
  else state-based estimate if user state exists
  else user default tax rate
- Distance filtering:
  same city
  same state
  same country
  any
- Brand filter:
  strict if selected
- Rating filter:
  positive ratings >= `minPositive` default `5`
  negative ratings <= `max(1, floor(positive / 5))`
- Free shipping filter.
- Seller standing:
  ranking boost, not a hard requirement by default
- Facebook special case:
  flag and exclude `free`, `$1`, `1$` placeholders.
- Offer detection:
  flag any result containing `offer` or `best offer`.

## Phase 7: Rank And Display
- Group by platform.
- Sort primarily by `totalCost`.
- Secondary ranking:
  better seller standing boost
  better condition
  closer distance
  better seller rating ratio
- Optional coupon savings if coupon lookup is enabled later
- Show visible flags for:
  placeholder price
  description-price override
  offer / best offer
  missing shipping/tax data

## Data Model
```js
{
  sourceListing: {
    platform: 'ebay' | 'facebook' | 'craigslist',
    title: '',
    description: '',
    query: '',
    brand: '',
    extractedAt: 0
  },
  filters: {
    distanceScope: 'city' | 'state' | 'country' | 'any',
    sameLocationValue: '',
    brandRequired: false,
    brand: '',
    freeShippingOnly: false,
    minPositiveRatings: 5,
    maxNegativeRatioDivisor: 5,
    sellerStandingBoost: true,
    userState: '',
    defaultTaxRate: 0,
    couponOptIn: false
  },
  consent: {
    cookiesPrompted: false,
    cookiesAllowed: false,
    historyAllowed: false,
    couponLookupAllowed: false
  },
  history: [
    {
      type: 'view' | 'search',
      platform: '',
      title: '',
      query: '',
      url: '',
      createdAt: 0
    }
  },
  results: [
    {
      platform: '',
      title: '',
      url: '',
      listedPrice: 0,
      descriptionPriceHint: null,
      effectivePrice: 0,
      shipping: 0,
      taxes: 0,
      totalCost: 0,
      condition: '',
      sellerName: '',
      sellerStanding: '',
      positiveRatings: 0,
      negativeRatings: 0,
      distanceText: '',
      city: '',
      state: '',
      country: '',
      flags: []
    }
  ]
}
```

## Per-Platform Notes
- eBay:
  usually the richest structured data source for price, shipping, condition, and seller rating.
- Facebook Marketplace:
  highest placeholder-price risk; treat `free` and `$1` variants as invalid comparison prices.
- Craigslist:
  weaker seller metadata; distance/location normalization matters more than ratings.

## Consent And Privacy
- Prompt once on first use before logging viewed listings or searches.
- If consent is denied, the extension should still function for manual comparison.
- Coupon lookup must be a separate opt-in even if basic history is allowed.

## Risk Areas
- DOM breakage due to marketplace UI changes.
- Marketplace policies around automation.
- Tax visibility may be inconsistent before checkout.
- Facebook result extraction may be sparse without user-visible navigation.

## Suggested Build Order
1. eBay extractor and result collector
2. query normalization
3. consent + persistent storage flow
4. Firefox popup/history UI
5. Facebook extractor with placeholder-price rules
6. Craigslist extractor
7. popup result table
8. ranking and filter polish
9. optional coupon adapters

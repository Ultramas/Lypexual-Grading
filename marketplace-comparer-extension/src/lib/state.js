export const DEFAULT_DRAFT = {
  brand: '',
  title: '',
  description: '',
};

export const DEFAULT_FILTERS = {
  distanceScope: 'any',
  brandRequired: false,
  brand: '',
  freeShippingOnly: false,
  sellerStandingBoost: true,
  userState: '',
  defaultTaxRate: 0,
  couponOptIn: false,
};

export const DEFAULT_CONSENT = {
  cookiesPrompted: false,
  cookiesAllowed: false,
  historyAllowed: false,
  couponLookupAllowed: false,
};

export function createEmptySessionState() {
  return {
    draft: { ...DEFAULT_DRAFT },
    filters: { ...DEFAULT_FILTERS },
    consent: { ...DEFAULT_CONSENT },
    sourceListing: null,
    results: [],
    history: [],
  };
}

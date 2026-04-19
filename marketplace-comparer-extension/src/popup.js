const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const brandInput = document.getElementById('brand');
const distanceScopeInput = document.getElementById('distanceScope');
const userStateInput = document.getElementById('userState');
const freeShippingOnlyInput = document.getElementById('freeShippingOnly');
const brandRequiredInput = document.getElementById('brandRequired');
const sellerStandingBoostInput = document.getElementById('sellerStandingBoost');
const couponOptInInput = document.getElementById('couponOptIn');
const consentCard = document.getElementById('consentCard');
const resultsNode = document.getElementById('results');
const resultsSummaryNode = document.getElementById('resultsSummary');
const historySummaryNode = document.getElementById('historySummary');
const statusPillsNode = document.getElementById('statusPills');

const FILTER_DEFAULTS = {
  distanceScope: 'any',
  sameLocationValue: '',
  brandRequired: false,
  brand: '',
  freeShippingOnly: false,
  minPositiveRatings: 5,
  maxNegativeRatioDivisor: 5,
  sellerStandingBoost: true,
  userState: '',
  defaultTaxRate: 0,
  couponOptIn: false,
};

document.getElementById('captureListing').addEventListener('click', captureCurrentListing);
document.getElementById('searchOtherPlatforms').addEventListener('click', searchOtherPlatforms);
document.getElementById('collectResults').addEventListener('click', collectCurrentResults);
document.getElementById('applyFilters').addEventListener('click', applyFilters);
document.getElementById('allowCookies').addEventListener('click', () => saveConsent(true));
document.getElementById('declineCookies').addEventListener('click', () => saveConsent(false));

bootstrap();

async function bootstrap() {
  const { draft, filters, consent, history, results, settings } = await chrome.storage.local.get([
    'draft',
    'filters',
    'consent',
    'history',
    'results',
    'settings',
  ]);

  restoreDraft(draft);
  restoreFilters(filters, settings);
  updateConsentUI(consent);
  renderStatusPills(filters, consent);
  renderHistorySummary(history || []);
  renderResultsSummary(results || []);

  if (history?.length) {
    render({ recentHistory: history.slice(0, 5) });
  }
}

async function captureCurrentListing() {
  const tab = await getActiveTab();
  if (!isSupportedMarketplaceUrl(tab?.url)) {
    render({ error: 'Open an eBay, Facebook Marketplace, or Craigslist page first.' });
    return;
  }

  let response;
  try {
    response = await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_LISTING' });
  } catch {
    render({ error: 'Could not reach the page script. Refresh the marketplace tab and try again.' });
    return;
  }

  if (!response?.title || !response?.description) {
    render({ error: 'Capture did not return both title and description yet. Add real selectors in src/content.js first.' });
    return;
  }

  if (response?.title) titleInput.value = response.title;
  if (response?.description) descriptionInput.value = response.description;

  await chrome.storage.local.set({ sourceListing: response, results: [] });
  renderResultsSummary([]);
  await persistDraft();
  await maybeSaveHistory({
    type: 'view',
    platform: response?.platform || 'unknown',
    title: response?.title || '',
    query: buildDraftQuery(),
    url: response?.url || tab.url || '',
  });
  renderStatusPills(await getSavedFilters(), await getSavedConsent());
  render(response);
}

async function searchOtherPlatforms() {
  const title = titleInput.value.trim();
  const description = descriptionInput.value.trim();
  const tab = await getActiveTab();

  if (!title || !description) {
    render({ error: 'Title and description are both required.' });
    return;
  }

  const query = buildDraftQuery();
  const activePlatform = detectPlatformFromUrl(tab?.url);
  const targets = await chrome.runtime.sendMessage({ type: 'BUILD_SEARCH_TARGETS', query });
  const filteredTargets = Object.fromEntries(
    Object.entries(targets).filter(([platform]) => platform !== activePlatform)
  );

  await chrome.storage.local.set({ results: [] });
  renderResultsSummary([]);
  await persistDraft();
  await maybeSaveHistory({
    type: 'search',
    platform: 'multi',
    title,
    query,
    url: '',
  });
  await Promise.all(Object.values(filteredTargets).map((url) => chrome.tabs.create({ url })));
  renderStatusPills(await getSavedFilters(), await getSavedConsent());
  render({ query, targets: filteredTargets, excludedPlatform: activePlatform || null });
}

async function collectCurrentResults() {
  const tab = await getActiveTab();
  if (!isSupportedMarketplaceUrl(tab?.url)) {
    render({ error: 'Open a supported marketplace results page first.' });
    return;
  }

  let response;
  try {
    response = await chrome.tabs.sendMessage(tab.id, { type: 'COLLECT_RESULTS' });
  } catch {
    render({ error: 'Could not reach the page script. Refresh the marketplace tab and try again.' });
    return;
  }

  const current = await chrome.storage.local.get(['results']);
  const results = [...(current.results || []), ...(response.results || [])];
  await chrome.storage.local.set({ results });
  renderResultsSummary(results);
  render({ collected: response.results?.length || 0, notes: response.notes || [] });
}

async function applyFilters() {
  const saved = await getSavedFilters();
  const { settings } = await chrome.storage.local.get(['settings']);
  const filters = {
    ...FILTER_DEFAULTS,
    ...(settings ? {
      minPositiveRatings: settings.minPositiveRatings ?? FILTER_DEFAULTS.minPositiveRatings,
      maxNegativeRatioDivisor: settings.maxNegativeRatioDivisor ?? FILTER_DEFAULTS.maxNegativeRatioDivisor,
      defaultTaxRate: settings.defaultTaxRate ?? FILTER_DEFAULTS.defaultTaxRate,
      userState: settings.defaultState ?? FILTER_DEFAULTS.userState,
    } : {}),
    ...saved,
    distanceScope: distanceScopeInput.value,
    brandRequired: brandRequiredInput.checked,
    brand: brandInput.value.trim(),
    freeShippingOnly: freeShippingOnlyInput.checked,
    sellerStandingBoost: sellerStandingBoostInput.checked,
    userState: userStateInput.value.trim() || settings?.defaultState || '',
    couponOptIn: couponOptInInput.checked,
  };

  await chrome.storage.local.set({ filters });
  const { consent } = await chrome.storage.local.get(['consent']);
  if (consent?.cookiesAllowed) {
    await chrome.storage.local.set({
      consent: {
        ...consent,
        couponLookupAllowed: couponOptInInput.checked,
      },
    });
  }
  await persistDraft();
  renderStatusPills(filters, consent);
  render({ message: 'Saved filter defaults. Implement filtering in src/lib/filters.js.' });
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function render(value) {
  resultsNode.textContent = JSON.stringify(value, null, 2);
}

function renderResultsSummary(results = []) {
  if (!results.length) {
    resultsSummaryNode.innerHTML = `<div class="miniItem"><strong>No ranked results yet</strong><div class="miniMeta">This section will later show best landed-cost matches across platforms.</div></div>`;
    return;
  }

  resultsSummaryNode.innerHTML = results.slice(0, 3).map((result) => `
    <div class="miniItem">
      <strong>${escapeHtml(result.title || 'Untitled result')}</strong>
      <div class="miniMeta">${escapeHtml(result.platform || 'unknown')} · $${Number(result.listedPrice || 0).toFixed(2)} listed</div>
    </div>
  `).join('');
}

function renderHistorySummary(history = []) {
  if (!history.length) {
    historySummaryNode.innerHTML = `<div class="miniItem"><strong>No history yet</strong><div class="miniMeta">Searches and viewed listings only appear here after consent.</div></div>`;
    return;
  }

  historySummaryNode.innerHTML = history.slice(0, 4).map((entry) => `
    <div class="miniItem">
      <strong>${escapeHtml(entry.title || entry.query || 'Untitled action')}</strong>
      <div class="miniMeta">${escapeHtml(entry.platform || 'unknown')} · ${escapeHtml(entry.type || 'event')}</div>
    </div>
  `).join('');
}

function renderStatusPills(filters = {}, consent = {}) {
  const pills = [
    'Draft Ready',
    'Manual Capture Flow',
    'Firefox First',
    consent?.historyAllowed ? 'History Enabled' : 'History Off',
    filters?.couponOptIn ? 'Coupon Opt-In' : 'Coupons Off',
    filters?.sellerStandingBoost !== false ? 'Seller Boost On' : 'Seller Boost Off',
  ];

  statusPillsNode.innerHTML = pills.map((pill) => `<span class="pill">${escapeHtml(pill)}</span>`).join('');
}

async function persistDraft() {
  const draft = {
    brand: brandInput.value.trim(),
    title: titleInput.value.trim(),
    description: descriptionInput.value.trim(),
  };

  await chrome.storage.local.set({ draft });
}

function restoreDraft(draft = {}) {
  brandInput.value = draft?.brand || '';
  titleInput.value = draft?.title || '';
  descriptionInput.value = draft?.description || '';
}

function restoreFilters(filters = {}, settings = {}) {
  const merged = {
    ...FILTER_DEFAULTS,
    ...(settings ? {
      minPositiveRatings: settings.minPositiveRatings ?? FILTER_DEFAULTS.minPositiveRatings,
      maxNegativeRatioDivisor: settings.maxNegativeRatioDivisor ?? FILTER_DEFAULTS.maxNegativeRatioDivisor,
      defaultTaxRate: settings.defaultTaxRate ?? FILTER_DEFAULTS.defaultTaxRate,
      userState: settings.defaultState ?? FILTER_DEFAULTS.userState,
    } : {}),
    ...(filters || {}),
  };

  distanceScopeInput.value = merged.distanceScope;
  userStateInput.value = merged.userState;
  freeShippingOnlyInput.checked = Boolean(merged.freeShippingOnly);
  brandRequiredInput.checked = Boolean(merged.brandRequired);
  sellerStandingBoostInput.checked = merged.sellerStandingBoost !== false;
  couponOptInInput.checked = Boolean(merged.couponOptIn);
}

async function saveConsent(allowed) {
  const consent = {
    cookiesPrompted: true,
    cookiesAllowed: allowed,
    historyAllowed: allowed,
    couponLookupAllowed: allowed && couponOptInInput.checked,
  };

  await chrome.storage.local.set({ consent });
  updateConsentUI(consent);
  renderStatusPills(await getSavedFilters(), consent);
  render({ consent });
}

function updateConsentUI(consent = {}) {
  const prompted = Boolean(consent?.cookiesPrompted);
  consentCard.style.display = prompted ? 'none' : 'grid';
}

async function maybeSaveHistory(entry) {
  const response = await chrome.runtime.sendMessage({ type: 'SAVE_HISTORY', entry });
  renderHistorySummary(response?.history || []);
}

function buildDraftQuery() {
  return [brandInput.value.trim(), titleInput.value.trim(), descriptionInput.value.trim()]
    .filter(Boolean)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function getSavedFilters() {
  const { filters } = await chrome.storage.local.get(['filters']);
  return filters || {};
}

async function getSavedConsent() {
  const { consent } = await chrome.storage.local.get(['consent']);
  return consent || {};
}

function isSupportedMarketplaceUrl(url = '') {
  return detectPlatformFromUrl(url) !== 'unknown';
}

function detectPlatformFromUrl(url = '') {
  if (url.includes('ebay.com')) return 'ebay';
  if (url.includes('facebook.com')) return 'facebook';
  if (url.includes('craigslist.org')) return 'craigslist';
  return 'unknown';
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

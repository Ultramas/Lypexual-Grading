(function bootstrap() {
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === 'CAPTURE_LISTING') {
      sendResponse(captureCurrentListing());
      return;
    }

    if (message?.type === 'COLLECT_RESULTS') {
      sendResponse(collectCurrentResults());
      return;
    }
  });
})();

function captureCurrentListing() {
  const platform = detectPlatform(window.location.hostname);

  return {
    platform,
    url: window.location.href,
    title: '',
    description: '',
    listedPrice: null,
    shipping: null,
    taxes: null,
    condition: '',
    sellerName: '',
    sellerStanding: '',
    positiveRatings: null,
    negativeRatings: null,
    locationText: '',
    bestOfferDetected: false,
    placeholderPriceFlag: false,
    notes: [`Add ${platform} selectors in src/content.js`],
  };
}

function collectCurrentResults() {
  const platform = detectPlatform(window.location.hostname);

  return {
    platform,
    results: [],
    notes: [`Add result-card selectors for ${platform} in src/content.js`],
  };
}

function detectPlatform(hostname) {
  if (hostname.includes('ebay.com')) return 'ebay';
  if (hostname.includes('facebook.com')) return 'facebook';
  if (hostname.includes('craigslist.org')) return 'craigslist';
  return 'unknown';
}

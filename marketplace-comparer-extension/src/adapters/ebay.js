export function getEbayAdapter() {
  return {
    platform: 'ebay',
    buildSearchUrl(query) {
      return `https://www.ebay.com/sch/i.html?_nkw=${encodeURIComponent(query)}`;
    },
    captureListing() {
      return {
        platform: 'ebay',
        supported: false,
        notes: ['Implement eBay listing selectors here.'],
      };
    },
    collectResults() {
      return {
        platform: 'ebay',
        supported: false,
        results: [],
        notes: ['Implement eBay search result selectors here.'],
      };
    },
  };
}

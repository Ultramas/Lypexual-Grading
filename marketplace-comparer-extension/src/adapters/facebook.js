export function getFacebookAdapter() {
  return {
    platform: 'facebook',
    buildSearchUrl(query) {
      return `https://www.facebook.com/marketplace/search/?query=${encodeURIComponent(query)}`;
    },
    captureListing() {
      return {
        platform: 'facebook',
        supported: false,
        notes: ['Implement Facebook Marketplace listing selectors here.'],
      };
    },
    collectResults() {
      return {
        platform: 'facebook',
        supported: false,
        results: [],
        notes: ['Implement Facebook Marketplace result selectors here.'],
      };
    },
  };
}

export function getCraigslistAdapter() {
  return {
    platform: 'craigslist',
    buildSearchUrl(query) {
      return `https://www.craigslist.org/search/sss?query=${encodeURIComponent(query)}`;
    },
    captureListing() {
      return {
        platform: 'craigslist',
        supported: false,
        notes: ['Implement Craigslist listing selectors here.'],
      };
    },
    collectResults() {
      return {
        platform: 'craigslist',
        supported: false,
        results: [],
        notes: ['Implement Craigslist result selectors here.'],
      };
    },
  };
}

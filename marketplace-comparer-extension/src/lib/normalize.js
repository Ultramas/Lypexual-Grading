export function buildQuery({ brand, title, description }) {
  return [brand, title, description]
    .filter(Boolean)
    .map((part) => String(part).trim())
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function detectOfferLanguage(text) {
  return /\bbest offer\b|\boffer\b/i.test(String(text || ''));
}

export function extractMoneyHints(text) {
  const matches = String(text || '').match(/\$\s?\d+(?:\.\d{1,2})?/g) || [];
  return matches.map((value) => Number(value.replace(/[^\d.]/g, ''))).filter(Number.isFinite);
}

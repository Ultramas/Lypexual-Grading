export function isFacebookPlaceholderPrice({ platform, listedPrice, title, description }) {
  if (platform !== 'facebook') return false;

  const text = `${title || ''} ${description || ''}`.toLowerCase();
  if (text.includes('free')) return true;
  return listedPrice === 1;
}

export function computeNegativeLimit(positiveRatings, divisor = 5) {
  return Math.max(1, Math.floor((Number(positiveRatings) || 0) / divisor));
}

export function passesSellerRating(result, { minPositiveRatings = 5, maxNegativeRatioDivisor = 5 } = {}) {
  const positive = Number(result.positiveRatings || 0);
  const negative = Number(result.negativeRatings || 0);
  return positive >= minPositiveRatings && negative <= computeNegativeLimit(positive, maxNegativeRatioDivisor);
}

export function computeTotalCost({ effectivePrice = 0, shipping = 0, taxes = 0 }) {
  return Number(effectivePrice || 0) + Number(shipping || 0) + Number(taxes || 0);
}

export function computeTaxEstimate({ listedTax, effectivePrice = 0, defaultTaxRate = 0, stateTaxRate = null }) {
  if (Number.isFinite(Number(listedTax)) && Number(listedTax) > 0) {
    return Number(listedTax);
  }

  const rate = Number.isFinite(Number(stateTaxRate)) ? Number(stateTaxRate) : Number(defaultTaxRate || 0);
  return Number(effectivePrice || 0) * rate;
}

export function computeRankingBoosts(result, { sellerStandingBoost = true, couponSavings = 0 } = {}) {
  let boost = 0;

  if (sellerStandingBoost && result?.sellerStanding) {
    boost += 10;
  }

  if (couponSavings > 0) {
    boost += Math.min(15, couponSavings);
  }

  return boost;
}

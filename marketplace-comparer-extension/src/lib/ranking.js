import { computeRankingBoosts, computeTaxEstimate, computeTotalCost } from './filters.js';

export function buildComparableResult(result, options = {}) {
  const effectivePrice = Number(result.descriptionPriceHint || result.listedPrice || 0);
  const taxes = computeTaxEstimate({
    listedTax: result.taxes,
    effectivePrice,
    defaultTaxRate: options.defaultTaxRate,
    stateTaxRate: options.stateTaxRate,
  });
  const totalCost = computeTotalCost({ effectivePrice, shipping: result.shipping, taxes });
  const rankingBoost = computeRankingBoosts(result, options);

  return {
    ...result,
    effectivePrice,
    taxes,
    totalCost,
    rankingBoost,
    adjustedRankScore: totalCost - rankingBoost,
  };
}

export function rankResults(results = [], options = {}) {
  return results
    .map((result) => buildComparableResult(result, options))
    .sort((a, b) => a.adjustedRankScore - b.adjustedRankScore);
}

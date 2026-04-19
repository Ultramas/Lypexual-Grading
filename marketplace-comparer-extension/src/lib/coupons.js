export function createCouponLookupPlan({ enabled, query, platform }) {
  if (!enabled) {
    return {
      enabled: false,
      providers: [],
      notes: ['Coupon lookup disabled by user.'],
    };
  }

  return {
    enabled: true,
    query,
    platform,
    providers: ['retailmenot', 'manual coupon provider', 'future adapter slot'],
    notes: ['Implement provider adapters and permission review before live coupon search.'],
  };
}

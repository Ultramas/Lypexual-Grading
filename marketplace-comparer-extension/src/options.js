const minPositiveRatingsNode = document.getElementById('minPositiveRatings');
const maxNegativeRatioDivisorNode = document.getElementById('maxNegativeRatioDivisor');
const defaultTaxRateNode = document.getElementById('defaultTaxRate');
const defaultStateNode = document.getElementById('defaultState');
const statusNode = document.getElementById('status');

document.getElementById('save').addEventListener('click', save);
restore();

async function save() {
  const settings = {
    minPositiveRatings: Number(minPositiveRatingsNode.value || 5),
    maxNegativeRatioDivisor: Number(maxNegativeRatioDivisorNode.value || 5),
    defaultTaxRate: Number(defaultTaxRateNode.value || 0),
    defaultState: String(defaultStateNode.value || '').trim(),
  };

  await chrome.storage.local.set({ settings });
  statusNode.textContent = JSON.stringify(settings, null, 2);
}

async function restore() {
  const { settings } = await chrome.storage.local.get('settings');
  if (!settings) return;

  minPositiveRatingsNode.value = settings.minPositiveRatings ?? 5;
  maxNegativeRatioDivisorNode.value = settings.maxNegativeRatioDivisor ?? 5;
  defaultTaxRateNode.value = settings.defaultTaxRate ?? 0;
  defaultStateNode.value = settings.defaultState ?? '';
}

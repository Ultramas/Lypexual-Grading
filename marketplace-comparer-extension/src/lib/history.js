export function createHistoryEntry({ type, platform, title = '', query = '', url = '' }) {
  return {
    type,
    platform,
    title,
    query,
    url,
    createdAt: Date.now(),
  };
}

export function formatHistoryLabel(entry) {
  const main = entry?.title || entry?.query || 'Untitled action';
  const platform = entry?.platform ? `[${entry.platform}] ` : '';
  return `${platform}${entry?.type || 'event'} · ${main}`;
}

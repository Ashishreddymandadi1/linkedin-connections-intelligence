const KEY = "lci.currentDataset";

export function setCurrentDataset(id: string) {
  localStorage.setItem(KEY, id);
}

export function getCurrentDataset(): string | null {
  return localStorage.getItem(KEY);
}

export function clearCurrentDataset() {
  localStorage.removeItem(KEY);
}

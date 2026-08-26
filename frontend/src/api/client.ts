import type {
  DatasetStatus,
  DatasetSummary,
  PersonListItem,
  SearchResponse,
  UploadReport,
} from "./types";

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<Record<string, unknown>>("/health"),

  listDatasets: () => req<DatasetSummary[]>("/datasets"),

  getDataset: (id: string) => req<DatasetSummary>(`/datasets/${id}`),

  uploadCsv: (file: File, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return req<UploadReport>("/datasets", { method: "POST", body: fd });
  },

  deleteDataset: (id: string) => req<void>(`/datasets/${id}`, { method: "DELETE" }),

  datasetStatus: (id: string) => req<DatasetStatus>(`/datasets/${id}/status`),

  startEnrichment: (id: string) =>
    req<{ started: boolean; job_id?: string; pending: number; message?: string }>(
      `/datasets/${id}/enrich`,
      { method: "POST" },
    ),

  listPeople: (id: string) => req<PersonListItem[]>(`/datasets/${id}/people`),

  search: (datasetId: string, query: string) =>
    req<SearchResponse>("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, query }),
    }),

  getSearch: (searchId: string) => req<SearchResponse>(`/search/${searchId}`),
};

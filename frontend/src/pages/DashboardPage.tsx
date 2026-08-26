import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Search as SearchIcon, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { Button, Card, StatCard } from "../components/ui";
import { clearCurrentDataset } from "../store";

export default function DashboardPage() {
  const { datasetId = "" } = useParams();
  const nav = useNavigate();

  const status = useQuery({
    queryKey: ["status", datasetId],
    queryFn: () => api.datasetStatus(datasetId),
    refetchInterval: (q) =>
      q.state.data && q.state.data.progress_done < q.state.data.progress_total ? 3000 : false,
  });

  const people = useQuery({
    queryKey: ["people", datasetId],
    queryFn: () => api.listPeople(datasetId),
  });

  const s = status.data;

  async function del() {
    if (!confirm("Delete this dataset and all enriched profiles, embeddings and search history?")) return;
    await api.deleteDataset(datasetId);
    clearCurrentDataset();
    nav("/upload");
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{s?.name ?? "Your Network"}</h1>
          <p className="text-sm text-ink-faint">
            Last updated {s?.last_updated ? new Date(s.last_updated).toLocaleString() : "—"}
          </p>
        </div>
        <Button variant="danger" onClick={del}>
          <span className="flex items-center gap-1.5">
            <Trash2 size={14} /> Delete
          </span>
        </Button>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Connections" value={s?.connections ?? "—"} />
        <StatCard label="Ready" value={s?.ready ?? "—"} tone="good" />
        <StatCard label="Partial" value={s?.partial ?? "—"} tone="warn" />
        <StatCard label="Failed" value={s?.failed ?? "—"} tone={s?.failed ? "bad" : "default"} />
      </div>

      {s && s.progress_done < s.progress_total && (
        <Link
          to={`/datasets/${datasetId}/enriching`}
          className="mt-4 block text-sm text-accent hover:underline"
        >
          Enrichment in progress — {s.progress_done}/{s.progress_total} →
        </Link>
      )}

      <div className="mt-8">
        <Button onClick={() => nav(`/datasets/${datasetId}/search`)}>
          <span className="flex items-center gap-2">
            <SearchIcon size={16} /> Search network
          </span>
        </Button>
      </div>

      <Card className="mt-8 divide-y divide-slate-100">
        <div className="px-5 py-3 text-xs font-semibold uppercase tracking-wide text-ink-faint">
          Connections {people.data ? `(${people.data.length})` : ""}
        </div>
        {people.data?.slice(0, 40).map((p) => (
          <div key={p.person_id} className="flex items-center justify-between px-5 py-2.5 text-sm">
            <div>
              <div className="font-medium">{p.full_name ?? p.linkedin_url}</div>
              <div className="text-ink-faint">
                {p.current_title ?? p.headline ?? "—"}
                {p.current_company ? ` · ${p.current_company}` : ""}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-ink-faint">{p.profile_completeness}/100</span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-ink-soft">
                {p.enrichment_state.toLowerCase().replace(/_/g, " ")}
              </span>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}

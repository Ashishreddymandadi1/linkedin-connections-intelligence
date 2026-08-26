import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { Button, Card, ProgressBar } from "../components/ui";
import { stateLabel } from "../lib/format";

export default function EnrichmentPage() {
  const { datasetId = "" } = useParams();
  const nav = useNavigate();

  const status = useQuery({
    queryKey: ["status", datasetId],
    queryFn: () => api.datasetStatus(datasetId),
    refetchInterval: 2000,
  });

  const start = useMutation({
    mutationFn: () => api.startEnrichment(datasetId),
    onSuccess: () => status.refetch(),
  });

  const s = status.data;
  const done = s ? s.progress_done >= s.progress_total && s.progress_total > 0 : false;

  useEffect(() => {
    // auto-kick enrichment once when nothing has run yet
    if (s && !s.job && s.pending > 0 && start.isIdle) start.mutate();
  }, [s, start]);

  return (
    <div className="mx-auto max-w-xl text-center">
      <h1 className="text-2xl font-semibold">Preparing your professional network</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Enriching each connection with LinkedIn profile data. You can leave this page — it resumes where it left off.
      </p>

      <Card className="mt-8 p-8">
        {!s ? (
          <p className="text-ink-faint">Loading…</p>
        ) : (
          <>
            <div className="font-mono text-3xl font-semibold">
              {s.progress_done} <span className="text-ink-faint">/ {s.progress_total}</span>
            </div>
            <div className="mt-4">
              <ProgressBar pct={s.progress_pct} />
            </div>
            <div className="mt-2 font-mono text-sm text-ink-soft">{s.progress_pct}%</div>

            <div className="mt-6 grid grid-cols-4 gap-2 text-xs">
              {(["ready", "partial", "failed", "waiting_for_llm"] as const).map((k) => (
                <div key={k} className="rounded-lg bg-slate-50 px-2 py-2">
                  <div className="font-mono text-lg font-semibold">{s[k]}</div>
                  <div className="text-ink-faint">{stateLabel(k)}</div>
                </div>
              ))}
            </div>

            {s.job?.error && <p className="mt-4 text-sm text-red-600">{s.job.error}</p>}

            <div className="mt-8 flex justify-center gap-3">
              {!done && (
                <Button variant="ghost" onClick={() => start.mutate()} disabled={start.isPending}>
                  {s.pending > 0 ? "Resume" : "Refresh"}
                </Button>
              )}
              <Button onClick={() => nav(`/datasets/${datasetId}`)} disabled={s.progress_total === 0}>
                {done ? "Go to dashboard" : "View dashboard"}
              </Button>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";
import { Button, Card } from "../components/ui";

const EXAMPLES = [
  "People who previously worked at Amazon",
  "People currently at Google who know Java",
  "Senior Java backend engineers",
  "People with AWS and distributed systems experience",
  "People who studied at Georgia Tech",
  "Who should I reach out to for AWS architecture advice?",
];

export default function SearchPage() {
  const { datasetId = "" } = useParams();
  const nav = useNavigate();
  const [q, setQ] = useState("");

  const run = useMutation({
    mutationFn: () => api.search(datasetId, q),
    onSuccess: (res) => nav(`/datasets/${datasetId}/search/${res.search_id}`, { state: res }),
  });

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-2xl font-semibold">Search your professional network</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Ask in plain English. Results rank the people you already know, backed by evidence from their profiles.
      </p>

      <Card className="mt-6 p-4">
        <textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          rows={3}
          placeholder="People who previously worked at Amazon and know AWS…"
          className="w-full resize-none rounded-lg border border-slate-200 p-3 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent-ring"
        />
        <div className="mt-3 flex justify-end">
          <Button type="submit" onClick={() => run.mutate()} disabled={q.trim().length < 3 || run.isPending}>
            {run.isPending ? "Searching…" : "Search"}
          </Button>
        </div>
      </Card>

      {run.error && <p className="mt-3 text-sm text-red-600">{String(run.error)}</p>}

      <div className="mt-6 flex flex-wrap gap-2">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            onClick={() => setQ(e)}
            className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-ink-soft hover:border-accent hover:text-accent"
          >
            {e}
          </button>
        ))}
      </div>
    </div>
  );
}

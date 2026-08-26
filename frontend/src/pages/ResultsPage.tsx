import { Link, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { SearchResponse } from "../api/types";
import { ResultCard } from "../components/ResultCard";
import { Badge, Card } from "../components/ui";

export default function ResultsPage() {
  const { datasetId = "", searchId = "" } = useParams();
  const location = useLocation();
  const preloaded = location.state as SearchResponse | undefined;

  const q = useQuery({
    queryKey: ["search", searchId],
    queryFn: () => api.getSearch(searchId),
    initialData: preloaded && preloaded.search_id === searchId ? preloaded : undefined,
  });

  const res = q.data;

  return (
    <div>
      <Link to={`/datasets/${datasetId}/search`} className="text-sm text-accent hover:underline">
        &larr; New search
      </Link>

      {!res ? (
        <p className="mt-6 text-ink-faint">Loading…</p>
      ) : (
        <>
          <h1 className="mt-3 text-xl font-semibold">&ldquo;{res.query}&rdquo;</h1>

          {res.interpreted_query?.criteria?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {res.interpreted_query.criteria.map((c) => (
                <Badge key={c.id} tone={c.required ? "accent" : "default"}>
                  {c.value} · {c.weight.toFixed(0)}
                  {c.required ? " · required" : ""}
                </Badge>
              ))}
            </div>
          )}

          <section className="mt-8">
            <div className="flex items-baseline gap-3">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-faint">
                Your connections
              </h2>
              <span className="text-xs text-ink-faint">
                {res.connections.returned} of {res.connections.total_candidates} candidates
              </span>
            </div>

            <div className="mt-4 space-y-3">
              {res.connections.results.map((item) => (
                <ResultCard key={item.person_id} item={item} />
              ))}
              {res.connections.results.length === 0 && (
                <Card className="px-5 py-8 text-center text-sm text-ink-faint">
                  No connections matched this query.
                </Card>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

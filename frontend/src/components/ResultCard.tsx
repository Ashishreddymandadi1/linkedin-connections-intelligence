import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import type { SearchResultItem } from "../api/types";
import { evidenceProvenance } from "../api/types";
import { initials } from "../lib/format";
import { Badge, ScoreMeter } from "./ui";

const PROVENANCE_LABEL: Record<string, string> = {
  linkedin: "LinkedIn data",
  ai_inferred: "AI inferred",
  company_inference: "Company inference",
  relevance: "Semantic relevance",
};
const PROVENANCE_TONE: Record<string, "fact" | "inferred"> = {
  linkedin: "fact",
  ai_inferred: "inferred",
  company_inference: "inferred",
  relevance: "inferred",
};

export function ResultCard({ item }: { item: SearchResultItem }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-slate-200 bg-card shadow-card">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-4 px-5 py-4 text-left"
      >
        <span className="font-mono text-sm text-ink-faint">#{item.rank}</span>
        {item.profile_picture_url ? (
          <img src={item.profile_picture_url} alt="" className="h-10 w-10 rounded-full object-cover" />
        ) : (
          <span className="grid h-10 w-10 place-items-center rounded-full bg-accent-soft text-sm font-semibold text-accent">
            {initials(item.name)}
          </span>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate font-semibold">{item.name ?? "Unknown"}</span>
            {item.is_connection && <Badge tone="accent">Connection</Badge>}
          </div>
          <div className="truncate text-sm text-ink-soft">
            {item.current_title ?? "—"}
            {item.current_company ? ` — ${item.current_company}` : ""}
            {item.location ? ` · ${item.location}` : ""}
          </div>
        </div>
        <div className="w-40 shrink-0">
          <ScoreMeter label="Match" value={item.match_score} emphasis />
        </div>
        <ChevronDown
          size={18}
          className={`shrink-0 text-ink-faint transition ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="space-y-5 border-t border-slate-100 px-5 py-5 text-sm">
          <div className="grid gap-4 sm:grid-cols-2">
            <ScoreMeter label="Match score" value={item.match_score} />
            <ScoreMeter label="Data confidence" value={item.data_confidence} />
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Why they match</div>
            <p className="mt-1 text-ink">{item.reason}</p>
          </div>

          {item.score_breakdown.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Score breakdown</div>
              <table className="mt-2 w-full font-mono text-xs">
                <tbody>
                  {item.score_breakdown.map((c) => (
                    <tr key={c.criterion_id} className="border-b border-slate-100 last:border-0">
                      <td className="py-1.5 pr-2">
                        {c.criterion}
                        {c.required && <span className="ml-1 text-accent">•req</span>}
                      </td>
                      <td className="py-1.5 text-right text-ink-faint">
                        {c.score.toFixed(0)}/{c.weight.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                  <tr>
                    <td className="pt-2 font-semibold">TOTAL</td>
                    <td className="pt-2 text-right font-semibold">{item.match_score.toFixed(0)}/100</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {item.evidence.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Evidence</div>
              <ul className="mt-1 space-y-1">
                {item.evidence.map((e, i) => {
                  const prov = evidenceProvenance(e.type);
                  const conf = e.detail?.confidence as number | undefined;
                  return (
                    <li key={i} className="flex gap-2">
                      <span className="text-ink-faint">•</span>
                      <span>
                        {e.text}{" "}
                        <Badge tone={PROVENANCE_TONE[prov]}>
                          {PROVENANCE_LABEL[prov]}
                          {prov !== "linkedin" && typeof conf === "number"
                            ? ` · ${Math.round(conf * 100)}%`
                            : ""}
                        </Badge>
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {item.relevant_skills.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {item.relevant_skills.map((s) => (
                <Badge key={s.skill_name} tone={s.is_inferred ? "inferred" : "fact"}>
                  {s.skill_name}
                  {s.is_inferred ? ` · ${Math.round(s.confidence * 100)}%` : ""}
                </Badge>
              ))}
            </div>
          )}

          <a
            href={item.linkedin_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-accent hover:underline"
          >
            View LinkedIn <ExternalLink size={13} />
          </a>
        </div>
      )}
    </div>
  );
}

import { Link, Outlet, useLocation } from "react-router-dom";
import { Network } from "lucide-react";
import { getCurrentDataset } from "./store";

export default function App() {
  const { pathname } = useLocation();
  const ds = getCurrentDataset();

  const nav = [
    { to: "/upload", label: "Upload" },
    ...(ds
      ? [
          { to: `/datasets/${ds}`, label: "Dashboard" },
          { to: `/datasets/${ds}/search`, label: "Search" },
        ]
      : []),
  ];

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-white">
              <Network size={16} />
            </span>
            Network Intelligence
          </Link>
          <nav className="flex gap-1 text-sm">
            {nav.map((n) => {
              const active = pathname === n.to || pathname.startsWith(n.to + "/");
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  className={`rounded-md px-3 py-1.5 transition ${
                    active
                      ? "bg-accent-soft text-accent"
                      : "text-ink-soft hover:bg-slate-100"
                  }`}
                >
                  {n.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}

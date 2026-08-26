import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, FileText } from "lucide-react";
import { api } from "../api/client";
import { setCurrentDataset } from "../store";
import { Button, Card } from "../components/ui";

export default function UploadPage() {
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const report = await api.uploadCsv(file, file.name.replace(/\.csv$/i, ""));
      setCurrentDataset(report.dataset.dataset_id);
      nav(`/datasets/${report.dataset.dataset_id}/enriching`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="text-2xl font-semibold">Upload your LinkedIn Connections CSV</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Export it from LinkedIn: <span className="font-medium">Settings &rsaquo; Data Privacy &rsaquo; Get a copy of your data &rsaquo; Connections</span>.
        Only the profile URL column is required.
      </p>

      <Card className="mt-6 p-2">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            const f = e.dataTransfer.files?.[0];
            if (f) setFile(f);
          }}
          className={`flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition ${
            drag ? "border-accent bg-accent-soft" : "border-slate-300 hover:border-accent"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <>
              <FileText className="text-accent" />
              <div className="font-medium">{file.name}</div>
              <div className="text-xs text-ink-faint">{(file.size / 1024).toFixed(0)} KB — click to replace</div>
            </>
          ) : (
            <>
              <UploadCloud className="text-ink-faint" />
              <div className="font-medium">Drop your Connections.csv here</div>
              <div className="text-xs text-ink-faint">or click to choose a file</div>
            </>
          )}
        </label>
      </Card>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      <div className="mt-5 flex justify-end">
        <Button onClick={submit} disabled={!file || busy}>
          {busy ? "Uploading…" : "Upload & continue"}
        </Button>
      </div>
    </div>
  );
}

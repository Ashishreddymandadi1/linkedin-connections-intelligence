from __future__ import annotations

import io
from pathlib import Path

from openpyxl import load_workbook

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def test_export_xlsx_has_four_sheets_and_profile_rows(client):
    ds_id = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}).json()[
        "dataset"
    ]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")

    r = client.get(f"/datasets/{ds_id}/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in r.headers["content-disposition"]
    assert r.headers["content-disposition"].endswith('.xlsx"')

    wb = load_workbook(io.BytesIO(r.content))
    assert wb.sheetnames == [
        "Profiles", "Experiences", "Education", "Skills",
        "Certifications", "Languages", "Publications",
    ]

    prof = wb["Profiles"]
    assert prof.max_row >= 6  # 5 fixtures + synthesized
    headers = [c.value for c in prof[1]]
    assert "LinkedIn URL" in headers and "Data Confidence" in headers

    names = {prof.cell(row=r_, column=1).value for r_ in range(2, prof.max_row + 1)}
    assert "Jane Smith" in names

    exp = wb["Experiences"]
    assert exp.max_row > 1
    assert any(
        exp.cell(row=r_, column=4).value == "Amazon" for r_ in range(2, exp.max_row + 1)
    )


def test_export_404_for_unknown_dataset(client):
    assert client.get("/datasets/nope/export").status_code == 404

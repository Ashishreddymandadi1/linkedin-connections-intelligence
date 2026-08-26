import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResultCard } from "./ResultCard";
import type { SearchResultItem } from "../api/types";

const item: SearchResultItem = {
  rank: 1,
  person_id: "p1",
  name: "Jane Smith",
  linkedin_url: "https://www.linkedin.com/in/jane-smith",
  profile_picture_url: null,
  current_title: "Senior Software Engineer",
  current_company: "Google",
  location: "Seattle, WA",
  is_connection: true,
  match_score: 94,
  data_confidence: 78,
  reason: "Jane previously worked at Amazon and lists AWS on her profile.",
  matched_criteria: ["Amazon", "AWS"],
  score_breakdown: [
    {
      criterion: "Previously at Amazon",
      criterion_id: "past_amazon",
      type: "past_company",
      weight: 60,
      match_strength: 1,
      score: 60,
      required: true,
      evidence: [],
    },
    {
      criterion: "AWS",
      criterion_id: "aws",
      type: "skill",
      weight: 40,
      match_strength: 0.85,
      score: 34,
      required: false,
      evidence: [],
    },
  ],
  evidence: [
    { type: "experience", text: "SDE II at Amazon (2021–2023)", detail: {} },
    { type: "semantic", text: "Microservices — inferred from ECS work", detail: {} },
  ],
  relevant_experience: [],
  relevant_skills: [],
  relevant_education: [],
};

describe("ResultCard", () => {
  it("shows match score and data confidence as distinct numbers", () => {
    render(<ResultCard item={item} />);
    expect(screen.getByText("Jane Smith")).toBeInTheDocument();
    expect(screen.getByText("Connection")).toBeInTheDocument();
    // collapsed header shows the match meter
    expect(screen.getAllByText("94").length).toBeGreaterThan(0);
  });

  it("expands to reveal score breakdown, evidence and fact/inference badges", () => {
    render(<ResultCard item={item} />);
    fireEvent.click(screen.getByText("Jane Smith"));

    expect(screen.getByText(/previously worked at Amazon/i)).toBeInTheDocument();
    expect(screen.getByText("Score breakdown")).toBeInTheDocument();
    expect(screen.getByText("60/60")).toBeInTheDocument();
    expect(screen.getByText("34/40")).toBeInTheDocument();
    expect(screen.getByText("94/100")).toBeInTheDocument();

    // both confidence and match meters present when expanded
    expect(screen.getByText("Data confidence")).toBeInTheDocument();
    expect(screen.getByText("78")).toBeInTheDocument();

    // fact vs inference distinction
    expect(screen.getByText("LinkedIn data")).toBeInTheDocument();
    expect(screen.getByText("AI inferred")).toBeInTheDocument();
  });
});

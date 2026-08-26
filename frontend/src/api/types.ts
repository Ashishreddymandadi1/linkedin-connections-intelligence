export interface DatasetSummary {
  dataset_id: string;
  name: string;
  connection_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface UploadReport {
  dataset: DatasetSummary;
  total_rows: number;
  imported: number;
  duplicates_removed: number;
  skipped_no_url: number;
  skipped: Array<Record<string, unknown>>;
  duplicates: Array<Record<string, unknown>>;
}

export interface DatasetStatus {
  dataset_id: string;
  name: string;
  status: string;
  connections: number;
  counts: Record<string, number>;
  ready: number;
  partial: number;
  failed: number;
  pending: number;
  waiting_for_llm: number;
  progress_done: number;
  progress_total: number;
  progress_pct: number;
  last_updated: string | null;
  job: null | {
    job_id: string;
    status: string;
    requested_profiles: number;
    completed_profiles: number;
    failed_profiles: number;
    error: string | null;
  };
}

export interface PersonListItem {
  person_id: string;
  full_name: string | null;
  headline: string | null;
  current_title: string | null;
  current_company: string | null;
  linkedin_url: string;
  profile_completeness: number;
  enrichment_state: string;
}

export interface EvidenceItem {
  type: string;
  text: string;
  detail: Record<string, unknown>;
}

export interface ScoreComponent {
  criterion: string;
  criterion_id: string;
  type: string;
  weight: number;
  match_strength: number;
  score: number;
  required: boolean;
  evidence: EvidenceItem[];
}

export interface ExperienceOut {
  position: string | null;
  company_name: string | null;
  company_linkedin_url: string | null;
  start_year: number | null;
  end_year: number | null;
  is_current: boolean;
  duration_text: string | null;
  description: string | null;
  location: string | null;
}

export interface EducationOut {
  school_name: string | null;
  degree: string | null;
  field_of_study: string | null;
  start_year: number | null;
  end_year: number | null;
}

export interface SkillOut {
  skill_name: string;
  source: string;
  is_inferred: boolean;
  confidence: number;
  evidence: string | null;
}

export interface SearchResultItem {
  rank: number;
  person_id: string;
  name: string | null;
  linkedin_url: string;
  profile_picture_url: string | null;
  current_title: string | null;
  current_company: string | null;
  location: string | null;
  is_connection: boolean;
  match_score: number;
  data_confidence: number;
  reason: string;
  matched_criteria: string[];
  score_breakdown: ScoreComponent[];
  evidence: EvidenceItem[];
  relevant_experience: ExperienceOut[];
  relevant_skills: SkillOut[];
  relevant_education: EducationOut[];
}

export interface SearchResponse {
  search_id: string;
  query: string;
  interpreted_query: {
    intent?: string;
    criteria: Array<{
      id: string;
      type: string;
      value: string;
      weight: number;
      required: boolean;
    }>;
  };
  connections: {
    total_candidates: number;
    returned: number;
    results: SearchResultItem[];
  };
  external: {
    searched: boolean;
    total_candidates: number;
    returned: number;
    results: SearchResultItem[];
  };
  llm_provider: string | null;
  llm_model: string | null;
}

export type Status = "MATCH_FOUND" | "UNCERTAIN" | "NO_RELIABLE_MATCH";

export interface Fingerprint {
  id: number;
  extractor: string;
  feature_version: string;
  dimension: number | null;
  summary: Record<string, unknown>;
  preprocessing: Record<string, unknown>;
  created_at: string;
}

export interface ReferenceImage {
  id: number;
  url: string;
  original_url: string;
  original_filename: string | null;
  width: number;
  height: number;
  capture_metadata: Record<string, string>;
  quality: Quality;
  uploaded_at: string;
  fingerprints: Fingerprint[];
}

export interface Quality {
  sharpness_laplacian_var?: number;
  brightness_mean?: number;
  contrast_std?: number;
  warnings?: string[];
  foreground_mask_reliable?: boolean;
  foreground_fraction?: number;
  embedding_message?: string;
}

export interface ObjectSummary {
  public_id: string;
  name: string;
  category: string;
  description: string | null;
  created_at: string;
  fingerprint_status: string;
  image_count: number;
  observation_count: number;
  thumbnail_url: string | null;
}

export interface ObjectDetail extends ObjectSummary {
  images: ReferenceImage[];
}

export interface ObjectList {
  items: ObjectSummary[];
  total: number;
  page: number;
  page_size: number;
  categories: string[];
}

export interface RegistrationResult {
  object: ObjectDetail;
  warnings: { filename: string | null; message: string }[];
}

export interface OrbMatch { qx: number; qy: number; rx: number; ry: number; inlier: boolean }

export interface Fused {
  score: number | null;
  normalized: Record<string, number | null>;
  contributions: Record<string, number>;
  used_weights: Record<string, number>;
  missing: string[];
}

export interface Evidence {
  raw: Record<string, number | null>;
  extras: Record<string, number>;
  orb?: {
    query_keypoints: number;
    reference_keypoints: number;
    good_matches: number;
    inliers: number;
    inlier_ratio: number;
    geometric_model: string | null;
  };
}

export interface Candidate {
  rank: number;
  object_public_id: string;
  name: string;
  category: string;
  score: number;
  best_reference_image_id: number;
  best_reference_url: string;
  supporting_references: number;
  references_compared: number;
  fused: Fused;
  evidence: Evidence;
  reference_scores: { image_id: number; score: number }[];
  matches: OrbMatch[];
}

export interface EmbeddingStatus {
  available: boolean;
  loaded?: boolean;
  backbone: string;
  version?: string;
  dimension?: number;
  description?: string;
  message?: string | null;
}

export interface FusionConfig {
  weights: Record<string, number>;
  ranges: Record<string, [number, number]>;
  accept_threshold: number;
  uncertain_threshold: number;
  margin_threshold: number;
  source: string;
  notes: string;
}

export interface IdentifyResponse {
  identification_id: number;
  status: Status;
  reason: string;
  predicted_object_id: string | null;
  predicted_name: string | null;
  predicted_category: string | null;
  best_score: number | null;
  margin: number | null;
  candidates: Candidate[];
  thresholds: { accept: number; uncertain: number; margin: number };
  fusion: FusionConfig;
  query_image_url: string;
  query_processed_size: [number, number];
  query_quality: Quality;
  inference_ms: number;
  timings: Record<string, number>;
  references_compared: number;
  embedding: EmbeddingStatus;
  note: string;
}

export interface CompareResponse {
  decision: "SAME_INSTANCE" | "DIFFERENT_INSTANCE" | "INCONCLUSIVE";
  reason: string;
  score: number | null;
  thresholds: { accept: number; uncertain: number };
  fused: Fused;
  evidence: Evidence;
  matches: OrbMatch[];
  image_a: string;
  image_b: string;
  size_a: [number, number];
  size_b: [number, number];
  quality_a: Quality;
  quality_b: Quality;
  raw_pixel_similarity: number;
  inference_ms: number;
  embedding: EmbeddingStatus;
  note: string;
}

export interface HistoryItem {
  id: number;
  created_at: string;
  status: Status;
  selected_object_public_id: string | null;
  best_score: number | null;
  inference_ms: number;
  query_image_url: string;
  top_candidates: { object_public_id: string; name: string; score: number }[];
  source: string;
}

export interface HistoryList { items: HistoryItem[]; total: number; page: number; page_size: number }

export interface Observation {
  id: number;
  observed_at: string;
  similarity: number;
  status: string;
  image_url: string;
  evidence: { fused?: Fused; evidence?: Evidence; decision?: string; reason?: string };
  surface_changes: Record<string, number | null>;
  note: string | null;
}

export interface ObservationResult {
  accepted: boolean;
  status: Status;
  reason: string;
  observation: Observation | null;
  score: number | null;
}

export interface ObjectHistory {
  object: ObjectSummary;
  first_registered: string;
  references: ReferenceImage[];
  observations: Observation[];
  note: string;
}

export interface Dashboard {
  usage: {
    registered_objects: number;
    reference_images: number;
    observations: number;
    query_attempts: number;
    accepted: number;
    uncertain: number;
    rejected: number;
    avg_inference_ms: number | null;
  };
  recent: HistoryItem[];
  categories: { category: string; count: number }[];
  model_status: EmbeddingStatus;
  fusion: FusionConfig;
  experiments: {
    run_uid: string;
    experiment: string;
    dataset_id: string;
    is_synthetic: boolean;
    created_at: string;
    identification_top1: Record<string, number | null>;
    verification_auc: Record<string, number | null>;
  }[];
  note: string;
}

export interface Health {
  status: string;
  database: boolean;
  storage: boolean;
  embedding: EmbeddingStatus;
  fusion_source: string;
  registered_objects: number;
  version: string;
}

export interface DatasetInfo { name: string; images: number; objects: number; synthetic: boolean; dataset_id: string }

export interface BinaryMetrics {
  threshold: number | null;
  tp: number; fp: number; tn: number; fn: number;
  precision: number | null; recall: number | null; f1: number | null;
  fmr: number | null; fnmr: number | null; accuracy: number | null;
  roc_auc?: number | null; eer?: number | null;
  curve?: { fpr: number; tpr: number }[];
}

export interface IdentMetrics {
  available: boolean;
  threshold: number;
  known_queries: number;
  unknown_queries: number;
  top1_accuracy: number | null;
  top5_accuracy: number | null;
  accepted: number;
  accepted_precision: number | null;
  accepted_recall: number | null;
  known_rejected_rate: number | null;
  misidentification_rate: number | null;
  unknown_rejection_rate: number | null;
  unknown_false_accept_rate: number | null;
  open_set_accuracy: number | null;
  matching_ms: number;
  extraction_ms: number | null;
  confusion?: { rows: string[]; cols: string[]; matrix: number[][] };
  validation?: Record<string, number | null>;
  threshold_selected_on: string;
}

export interface RobustnessCondition {
  condition: string;
  label: string;
  source: "dataset" | "perturbation";
  n_queries: number;
  status: "evaluated" | "pending";
  methods: Record<string, {
    top1_accuracy: number | null;
    open_set_accuracy: number | null;
    accepted_recall: number | null;
    unknown_rejection_rate: number | null;
    false_match_rate: number;
    mean_matching_ms: number;
  }>;
}

export interface ExperimentMetrics {
  run_uid?: string;
  dataset_id?: string;
  is_synthetic?: boolean;
  methods_evaluated?: string[];
  methods_unavailable?: Record<string, string>;
  method_labels?: Record<string, string>;
  embedding_backbone?: string | null;
  split?: Record<string, unknown> & {
    split_id: string; strategy: string; val_objects: string[]; test_objects: string[];
    known_objects: number; unknown_objects: string[];
    gallery: SplitPart; val_queries: SplitPart; test_queries: SplitPart;
    excluded: Record<string, number>; notes: string[];
  };
  hybrid_tuning?: { weights?: Record<string, number>; ranges?: Record<string, { lo: number; hi: number }>; validation_roc_auc?: number; message?: string };
  verification?: {
    val_pairs: Record<string, number | boolean>;
    test_pairs: Record<string, number | boolean>;
    methods: Record<string, { available: boolean; validation?: BinaryMetrics; test?: BinaryMetrics; threshold_selected_on?: string }>;
  };
  identification?: { gallery_images: number; known_objects: number; unknown_objects: number; latency_note: string; methods: Record<string, IdentMetrics> };
  robustness?: { conditions: RobustnessCondition[]; source: string };
  artifacts?: Record<string, string>;
  runtime_s?: number;
  error?: string;
}

export interface SplitPart { images: number; sessions: string[]; conditions: Record<string, number>; objects: number }

export interface ExperimentRun {
  run_uid: string;
  experiment: string;
  dataset_id: string;
  split_id: string | null;
  config: Record<string, unknown>;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string | null;
  metrics: ExperimentMetrics;
  artifact_paths: Record<string, string>;
  is_synthetic: boolean;
  created_at: string;
  finished_at: string | null;
}

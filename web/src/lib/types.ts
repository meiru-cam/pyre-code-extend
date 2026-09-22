export interface Test {
  name: string;
  code?: string;
  behavior?: string;
  visibility?: 'visible' | 'unshown';
  /** Unshown cases only: what the check looks for, phrased as the failure diagnosis. */
  failureMessage?: string;
  /** Unshown cases only: the evaluator code, revealed in the results pane after grading. */
  hiddenCode?: string;
}

export interface HintLevel {
  level: number;
  kind: 'questions' | 'analysis';
  content: string;
}

export interface SourceRef {
  kind: 'code' | 'paper';
  url: string;
  commit?: string;
  path?: string;
  symbol?: string;
  line?: number;
  section?: string;
  equation?: string;
  figure?: string;
  pages?: string;
  license?: string;
  adapted?: string;
  simplifications?: string;
}

export type DesignNoteFieldName =
  | 'api_boundaries'
  | 'state_ownership'
  | 'failure_recovery'
  | 'backpressure_concurrency'
  | 'durability_idempotency'
  | 'observability'
  | 'security'
  | 'tradeoffs';

export type DesignNoteFields = Record<DesignNoteFieldName, string>;

export interface DesignNoteResponse {
  taskId: string;
  contractVersion: number;
  fields: DesignNoteFields;
  updatedAt: string;
}

export interface Problem {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  functionName: string;
  hint: string;
  hintZh: string;
  descriptionEn: string;
  descriptionZh: string;
  tests: Test[];
  version?: number;
  hints?: HintLevel[];
  advisoryPrerequisites?: string[];
  modelConnections?: string[];
  proConAnalysis?: { pros: string[]; cons: string[] };
  designNoteRubric?: { field: DesignNoteFieldName; label: string }[];
  sources?: SourceRef[];
}

export interface TestResult {
  name: string;
  passed: boolean;
  execTimeMs: number;
  error?: string;
  output?: string;
  behavior?: string;
  visibility?: string;
  testIndex?: number;
  /** 1-based line the exception came from, within whichever body errorScope names. */
  errorLine?: number;
  errorLineText?: string;
  errorScope?: 'test' | 'solution';
}

export interface SubmissionResult {
  passed: number;
  total: number;
  allPassed: boolean;
  results: TestResult[];
  totalTimeMs: number;
  error?: string;
}

export interface ProblemProgress {
  status: 'todo' | 'attempted' | 'solved';
  bestTimeMs?: number;
  attempts: number;
  solvedAt?: string;
  contractVersion?: number;
  completedVersions?: number[];
}

export interface ProgressMap {
  [taskId: string]: ProblemProgress;
}

export interface CustomTest {
  name: string;
  code: string;
}

export interface LearningPath {
  id: string;
  titleEn: string;
  titleZh: string;
  descriptionEn: string;
  descriptionZh: string;
  icon: string;
  problems: string[];
  prerequisites: string[];
}

export interface LearningPathProblemSummary {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

export interface SubmissionHistory {
  id: number;
  passed: boolean;
  execTimeMs: number | null;
  submittedAt: string;
  code: string;
  contractVersion?: number;
  designNotePresent?: boolean;
}

export interface AiHelpResponse {
  guidance: string;
  model?: string;
  advisory?: boolean;
}

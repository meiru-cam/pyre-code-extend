import type { Locale } from '@/lib/i18n';

export interface Test {
  name: string;
  code?: string;
  behavior?: string;
  visibility?: 'visible' | 'unshown';
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
  designNoteRubric?: { field: string; label: string }[];
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
}

export interface AiHelpConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  includeUserCode: boolean;
}

export interface AiHelpRequest {
  problemId: string;
  problemTitle: string;
  functionName: string;
  description: string;
  solutionCode: string;
  sampleTests: Array<{ name: string; code: string }>;
  customPrompt?: string;
  userCode?: string;
  locale: Locale;
  config: Omit<AiHelpConfig, 'includeUserCode'>;
}

export interface AiHelpResponse {
  guidance: string;
  model?: string;
}

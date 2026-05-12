export type SetupStage =
  | 'checking'
  | 'installing-mlx'
  | 'starting-mlx'
  | 'downloading-model'
  | 'loading-model'
  | 'starting-server'
  | 'ready'
  | 'error'

export interface SetupStatus {
  stage: SetupStage
  message: string
  progress?: number
  bytesDone?: number
  bytesTotal?: number
  /** Per-file detail shown beneath the main message (file name, speed, etc.) */
  detail?: string
  error?: string
  /**
   * Set when the error is a stale/incompatible model cache.
   * The UI shows a "Clear cache & re-download" button instead of a plain retry.
   */
  cacheError?: boolean
}

export interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
  result?: string
  error?: string
  running?: boolean
}

export type Role = 'user' | 'assistant' | 'system' | 'tool'

export interface ChatMessage {
  id: string
  role: Role
  content: string
  toolCalls?: ToolCall[]
  createdAt: number
  model?: string
  done?: boolean
  activity?: AgentActivity
}

export type AgentMode = 'chat' | 'code'

export interface ChatRequest {
  conversationId: string
  messages: Array<{ role: Role; content: string; toolCalls?: ToolCall[] }>
  model: string
  enableTools: boolean
  mode: AgentMode
}

export interface WorkspaceInfo {
  conversationId: string
  path: string
  previewUrl: string
}

export interface WorkspaceFile {
  path: string
  kind: 'file' | 'dir'
  size?: number
}

export interface FileChangeEvent {
  conversationId: string
}

export type AgentActivity =
  | { kind: 'idle' }
  | { kind: 'thinking'; chars?: number }
  | { kind: 'generating'; chars?: number }
  | { kind: 'tool'; tool: string; target?: string; chars?: number }

export type StreamChunk =
  | { type: 'token'; text: string }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; id: string; result?: string; error?: string }
  | { type: 'activity'; activity: AgentActivity }
  | { type: 'done' }
  | { type: 'error'; error: string }

export interface ModelInfo {
  /** Model identifier — HuggingFace repo ID on macOS, Ollama tag on Windows */
  name: string
  /** Short, user-friendly display name */
  label: string
  size: string
  sizeBytes: number
  description: string
  recommended?: boolean
}

// ── macOS / MLX models ────────────────────────────────────────────────────────
// HuggingFace repo IDs for mlx-community quantised builds.

export const MLX_MODELS: ModelInfo[] = [
  {
    name: 'mlx-community/gemma-4-e2b-it-4bit',
    label: 'Gemma 4 E2B',
    size: '1.5 GB',
    sizeBytes: 1_500_000_000,
    description: 'Edge-sized. Fast & lightweight. Text + image + audio. Runs on 8 GB+ Macs.'
  },
  {
    name: 'mlx-community/gemma-4-e4b-it-4bit',
    label: 'Gemma 4 E4B',
    size: '3 GB',
    sizeBytes: 3_000_000_000,
    description: 'Best all-rounder. Text + image + audio. Runs on 8 GB+ Macs.',
    recommended: true
  },
  {
    name: 'mlx-community/gemma-4-26b-a4b-it-4bit',
    label: 'Gemma 4 27B MoE',
    size: '16 GB',
    sizeBytes: 16_000_000_000,
    description: 'Mixture-of-Experts (26B, 4B active). 16 GB+ RAM recommended.'
  },
  {
    name: 'mlx-community/gemma-4-31b-it-4bit',
    label: 'Gemma 4 31B',
    size: '18 GB',
    sizeBytes: 18_000_000_000,
    description: 'Frontier dense model. Best quality. 32 GB+ RAM recommended.'
  }
]

// ── Windows / Ollama models ───────────────────────────────────────────────────
// Ollama library tags — verify against https://ollama.com/library/gemma4

export const OLLAMA_MODELS: ModelInfo[] = [
  {
    name: 'gemma4:2b',
    label: 'Gemma 4 2B',
    size: '~1.7 GB',
    sizeBytes: 1_700_000_000,
    description: 'Fastest option. Great for everyday tasks. Runs on any PC with 8 GB+ RAM.'
  },
  {
    name: 'gemma4:4b',
    label: 'Gemma 4 4B',
    size: '~3.3 GB',
    sizeBytes: 3_300_000_000,
    description: 'Best all-rounder. Excellent quality and speed. Needs 8 GB+ RAM.',
    recommended: true
  },
  {
    name: 'gemma4:27b',
    label: 'Gemma 4 27B',
    size: '~17 GB',
    sizeBytes: 17_000_000_000,
    description: 'High quality. Best with a GPU; needs 32 GB+ RAM for CPU-only.'
  },
  {
    name: 'gemma4:31b',
    label: 'Gemma 4 31B',
    size: '~20 GB',
    sizeBytes: 20_000_000_000,
    description: 'Frontier quality. Requires a capable GPU or 32 GB+ RAM.'
  }
]

/** Backward-compat alias — new code should use MLX_MODELS or OLLAMA_MODELS. */
export const AVAILABLE_MODELS = MLX_MODELS

export const DEFAULT_MLX_MODEL = 'mlx-community/gemma-4-e4b-it-4bit'
export const DEFAULT_OLLAMA_MODEL = 'gemma4:4b'
/** Legacy alias — resolves to the MLX default. Renderer should use window.api.defaultModel. */
export const DEFAULT_MODEL = DEFAULT_MLX_MODEL


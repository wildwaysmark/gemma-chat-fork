/**
 * Ollama backend for Windows (and optionally Linux).
 *
 * Ollama exposes an OpenAI-compatible HTTP API on port 11434, so the chat
 * streaming and model-listing code from mlx.ts works unchanged.  This file
 * handles everything that IS different: locating the binary, starting/stopping
 * the server process, pulling models with progress feedback, and clearing them.
 *
 * Design principles (stability first):
 *  - Never kill an Ollama process we didn't spawn (user may have it as a
 *    system tray / menubar app).
 *  - Mirror the MLX backend's exported function signatures so index.ts can
 *    call them through simple platform branches.
 *  - Prefer the Ollama HTTP API over CLI where possible; fall back to CLI for
 *    operations that need the server stopped (e.g. rm).
 */

import { spawn, ChildProcess, spawnSync } from 'child_process'
import { homedir } from 'os'
import { join } from 'path'

const OLLAMA_PORT = 11434
const OLLAMA_URL = `http://127.0.0.1:${OLLAMA_PORT}`

// The ChildProcess we personally spawned (null if Ollama was already running).
let managedProc: ChildProcess | null = null
let currentModel: string | null = null

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export class OllamaNotFoundError extends Error {
  constructor() {
    super(
      'Ollama is not installed or not on PATH. ' +
      'Download it from https://ollama.com, install it, then restart the app.'
    )
    this.name = 'OllamaNotFoundError'
  }
}

// ---------------------------------------------------------------------------
// Progress type (structurally compatible with mlx.ts ServerProgress)
// ---------------------------------------------------------------------------

export interface OllamaProgress {
  stage?: 'starting-mlx' | 'downloading-model' | 'loading-model' | 'starting-server'
  message: string
  progress?: number
  bytesDone?: number
  bytesTotal?: number
  detail?: string
}

// ---------------------------------------------------------------------------
// Binary detection
// ---------------------------------------------------------------------------

/**
 * Search common installation locations for the ollama binary.
 * Returns the first path that responds to `ollama --version`, or null.
 */
export function findOllama(): string | null {
  const candidates: string[] =
    process.platform === 'win32'
      ? [
          // NSIS installer default
          join(
            process.env.LOCALAPPDATA ?? join(homedir(), 'AppData', 'Local'),
            'Programs',
            'Ollama',
            'ollama.exe'
          ),
          // Scoop / winget installs may put it on PATH
          'ollama.exe'
        ]
      : [
          // macOS pkg installer
          '/usr/local/bin/ollama',
          // Homebrew
          '/opt/homebrew/bin/ollama',
          // Generic PATH fallback
          'ollama'
        ]

  for (const c of candidates) {
    try {
      const r = spawnSync(c, ['--version'], {
        timeout: 5000,
        stdio: ['ignore', 'pipe', 'pipe']
      })
      if (r.status === 0) {
        console.log(`[ollama] Found binary: ${c} (${r.stdout?.toString().trim()})`)
        return c
      }
    } catch {
      // not available at this path
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// Health helpers
// ---------------------------------------------------------------------------

async function isServerRunning(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_URL}/v1/models`, {
      signal: AbortSignal.timeout(3000)
    })
    return res.ok
  } catch {
    return false
  }
}

async function waitForHealth(timeoutMs: number): Promise<void> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (await isServerRunning()) {
      console.log('[ollama] Server is healthy')
      return
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  throw new Error(
    `Ollama did not respond on port ${OLLAMA_PORT} within ${timeoutMs / 1000}s. ` +
    'Is Ollama installed and allowed through the firewall?'
  )
}

// ---------------------------------------------------------------------------
// Model pulling
// ---------------------------------------------------------------------------

interface OllamaPullEvent {
  status: string
  digest?: string
  total?: number
  completed?: number
  error?: string
}

/**
 * Stream-pull a model via the Ollama HTTP API and emit structured progress.
 * The API returns newline-delimited JSON objects with status + byte counts.
 */
async function pullModel(
  model: string,
  onProgress?: (p: OllamaProgress) => void
): Promise<void> {
  console.log(`[ollama] Pulling model: ${model}`)
  onProgress?.({ stage: 'downloading-model', message: `Starting download of ${model}…` })

  const res = await fetch(`${OLLAMA_URL}/api/pull`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, stream: true })
  })

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(
      `Ollama pull failed for ${model}: HTTP ${res.status} — ${text.slice(0, 200)}`
    )
  }

  const reader = (res.body as unknown as ReadableStream<Uint8Array>).getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let pullSucceeded = false

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })

      let newlineIdx: number
      while ((newlineIdx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, newlineIdx).trim()
        buf = buf.slice(newlineIdx + 1)
        if (!line || !onProgress) continue

        try {
          const ev = JSON.parse(line) as OllamaPullEvent

          if (ev.error) {
            throw new Error(`Ollama pull error: ${ev.error}`)
          }

          if (ev.status === 'success') {
            pullSucceeded = true
            onProgress({ stage: 'loading-model', message: 'Download complete — loading model…' })
            // BUG-03 fix: cancel the stream before returning so the body is
            // released and the connection can be reused by the runtime.
            reader.cancel().catch(() => {})
            return
          }

          // Byte-level progress (layer downloads)
          if (ev.total != null && ev.completed != null && ev.total > 0) {
            const pct = ev.completed / ev.total
            const shortDigest = ev.digest ? ev.digest.slice(7, 19) : ''
            onProgress({
              stage: 'downloading-model',
              message: `Downloading ${model} (${Math.round(pct * 100)}%)`,
              progress: pct,
              bytesDone: ev.completed,
              bytesTotal: ev.total,
              detail: shortDigest ? `Layer ${shortDigest}…` : undefined
            })
          } else if (ev.status) {
            // Status-only messages: "pulling manifest", "verifying sha256 digest", etc.
            onProgress({ stage: 'downloading-model', message: ev.status })
          }
        } catch (e) {
          // Rethrow actual errors (not just JSON parse failures)
          if ((e as Error).message.startsWith('Ollama pull error')) throw e
        }
      }
    }
  } finally {
    // Always release the reader even if we threw mid-stream.
    reader.cancel().catch(() => {})
  }

  // BUG-04 fix: if the stream closed without a 'success' event, Ollama likely
  // crashed mid-pull. Throw so the caller shows an error instead of silently
  // proceeding to verifyModelReady (which would hang for up to 600 s).
  if (!pullSucceeded) {
    throw new Error(
      `Ollama pull stream ended without a success status for ${model}. ` +
      'The server may have crashed — check the Ollama logs.'
    )
  }
}

// ---------------------------------------------------------------------------
// Model warm-up (same lazy-load pattern as MLX)
// ---------------------------------------------------------------------------

/**
 * Send a minimal chat completion request to trigger Ollama's lazy model load,
 * then wait for a real response.  Emits heartbeat progress every 3 s so the
 * UI elapsed timer keeps ticking during a silent load.
 */
async function verifyModelReady(
  model: string,
  timeoutMs: number,
  onProgress?: (p: OllamaProgress) => void
): Promise<void> {
  onProgress?.({ stage: 'loading-model', message: 'Loading model into memory…' })

  const controller = new AbortController()
  const heartbeatId = onProgress
    ? setInterval(
        () => onProgress({ stage: 'loading-model', message: 'Loading model into memory…' }),
        3000
      )
    : null
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(`${OLLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false,
        temperature: 0
      }),
      signal: controller.signal
    })

    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(
        `Model failed to load (HTTP ${res.status}): ${body.slice(0, 200)}`
      )
    }

    console.log('[ollama] Model warm-up complete — ready')
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      throw new Error(`Model did not finish loading within ${Math.round(timeoutMs / 1000)}s.`)
    }
    throw e
  } finally {
    clearTimeout(timeoutId)
    controller.abort()
    if (heartbeatId) clearInterval(heartbeatId)
  }
}

// ---------------------------------------------------------------------------
// Public API (mirrors mlx.ts exports used by index.ts)
// ---------------------------------------------------------------------------

export async function startServer(
  model: string,
  onProgress?: (p: OllamaProgress) => void
): Promise<void> {
  // Reuse an existing server we started if it's still healthy
  if (managedProc && !managedProc.killed && currentModel === model) {
    onProgress?.({ stage: 'starting-mlx', message: 'Connecting to Ollama…' })
    if (await isServerRunning()) {
      console.log('[ollama] Existing managed server is healthy, reusing')
      return
    }
  }

  await stopServer()

  const ollamaPath = findOllama()
  if (!ollamaPath) throw new OllamaNotFoundError()

  // If Ollama is already running as a system service / tray app, use it.
  const alreadyRunning = await isServerRunning()

  if (!alreadyRunning) {
    onProgress?.({ stage: 'starting-mlx', message: 'Starting Ollama server…' })
    console.log(`[ollama] Spawning: ${ollamaPath} serve`)

    const proc = spawn(ollamaPath, ['serve'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
      env: { ...process.env }
    })
    managedProc = proc

    proc.stdout?.on('data', (d: Buffer) =>
      console.log('[ollama stdout]', d.toString().trim())
    )
    proc.stderr?.on('data', (d: Buffer) =>
      console.log('[ollama stderr]', d.toString().trim())
    )
    // BUG-02 fix: capture `proc` in closure so a later re-spawn doesn't
    // accidentally null out `managedProc` for the *new* process.
    proc.on('exit', (code) => {
      console.log('[ollama] serve exited with code', code)
      if (managedProc === proc) {
        managedProc = null
        currentModel = null
      }
    })

    await waitForHealth(30_000)
  } else {
    console.log('[ollama] Using existing Ollama server')
    onProgress?.({ stage: 'starting-mlx', message: 'Ollama is running…' })
  }

  currentModel = model

  // Pull the model if it isn't already available locally
  const available = await listLocalModels()
  const modelReady = available.some(
    (m) => m === model || m === `${model}:latest` || m.startsWith(`${model}:`)
  )

  if (!modelReady) {
    await pullModel(model, onProgress)
  } else {
    console.log(`[ollama] Model ${model} already available`)
    onProgress?.({ stage: 'loading-model', message: 'Model already downloaded — loading…' })
  }

  // Warm-up: triggers lazy model load (same reason as MLX's verifyModelReady)
  const verifyTimeout = 600_000
  await verifyModelReady(model, verifyTimeout, onProgress)
}

/**
 * Stop the server if WE started it; leave system-service Ollama alone.
 * Waits for the process to exit so the port is free before the caller proceeds.
 */
export function stopServer(): Promise<void> {
  if (!managedProc || managedProc.killed) {
    managedProc = null
    currentModel = null
    return Promise.resolve()
  }

  const proc = managedProc
  managedProc = null
  currentModel = null
  console.log('[ollama] Stopping managed server — waiting for process to exit')

  return new Promise<void>((resolve) => {
    const killTimer = setTimeout(() => {
      console.log('[ollama] Process did not exit in 3 s — sending SIGKILL')
      try {
        proc.kill('SIGKILL')
      } catch { /* already dead */ }
      resolve()
    }, 3000)

    proc.once('exit', () => {
      clearTimeout(killTimer)
      console.log('[ollama] Managed server exited cleanly')
      resolve()
    })

    proc.kill('SIGTERM')
  })
}

/**
 * Remove a model from Ollama's local storage.
 * Uses the CLI so it works even when the HTTP server is stopped.
 */
export function clearModelCache(model: string): void {
  const ollamaPath = findOllama()
  if (!ollamaPath) {
    console.warn('[ollama] clearModelCache: ollama binary not found, skipping')
    return
  }
  console.log(`[ollama] Removing model: ${model}`)
  const result = spawnSync(ollamaPath, ['rm', model], {
    stdio: ['ignore', 'pipe', 'pipe'],
    timeout: 30_000
  })
  if (result.status === 0) {
    console.log(`[ollama] Removed model: ${model}`)
  } else {
    console.warn(
      `[ollama] Failed to remove ${model}: ${result.stderr?.toString().trim()}`
    )
  }
}

/**
 * List models currently stored in Ollama (OpenAI-compat endpoint).
 * Returns [] if the server is not running.
 */
export async function listLocalModels(): Promise<string[]> {
  try {
    const res = await fetch(`${OLLAMA_URL}/v1/models`, {
      signal: AbortSignal.timeout(3000)
    })
    if (!res.ok) return []
    const data = (await res.json()) as { data?: Array<{ id: string }> }
    return (data.data ?? []).map((m) => m.id)
  } catch {
    return []
  }
}

/**
 * Check whether the Ollama binary is findable AND the server is reachable.
 * Used by the setup:status IPC handler on Windows.
 */
export async function checkBackendReady(): Promise<boolean> {
  if (!findOllama()) return false
  return isServerRunning()
}

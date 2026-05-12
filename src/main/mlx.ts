import { app } from 'electron'
import { spawn, ChildProcess, spawnSync } from 'child_process'
import { join } from 'path'
import { existsSync, rmSync } from 'fs'

const MLX_PORT = 11434
const MLX_HOST = `127.0.0.1:${MLX_PORT}`
const MLX_URL = `http://${MLX_HOST}`

let serverProc: ChildProcess | null = null
let currentModel: string | null = null

// ---------------------------------------------------------------------------
// Paths — everything lives under <appData>/mlx/
// ---------------------------------------------------------------------------

function dataDir(): string {
  return join(app.getPath('userData'), 'mlx')
}

function venvDir(): string {
  return join(dataDir(), 'venv')
}

/** The python binary inside our managed venv */
function venvPython(): string {
  return join(venvDir(), 'bin', 'python3')
}

function modelsDir(): string {
  return join(dataDir(), 'models')
}

// ---------------------------------------------------------------------------
// System Python detection
// ---------------------------------------------------------------------------

/**
 * Find a compatible system Python (3.10–3.13).
 * We explicitly skip 3.14+ because mlx-lm doesn't publish wheels for it yet.
 * We try versioned binaries first (most reliable), then fall back to `python3`.
 */
function findSystemPython(): string | null {
  // Prefer specific known-good versions, newest first
  const versionedCandidates = [
    '/opt/homebrew/bin/python3.13',
    '/opt/homebrew/bin/python3.12',
    '/opt/homebrew/bin/python3.11',
    '/opt/homebrew/bin/python3.10',
    '/opt/homebrew/opt/python@3.13/bin/python3.13',
    '/opt/homebrew/opt/python@3.12/bin/python3.12',
    '/opt/homebrew/opt/python@3.11/bin/python3.11',
    '/opt/homebrew/opt/python@3.10/bin/python3.10',
    '/usr/local/bin/python3.13',
    '/usr/local/bin/python3.12',
    '/usr/local/bin/python3.11',
    '/usr/local/bin/python3.10'
  ]

  for (const c of versionedCandidates) {
    try {
      const s = spawnSync(c, ['--version'], { timeout: 5000, stdio: ['ignore', 'pipe', 'pipe'] })
      if (s.status === 0) {
        console.log(`[mlx] Found compatible Python: ${c} (${s.stdout.toString().trim()})`)
        return c
      }
    } catch {
      // not available
    }
  }

  // Last resort: try generic python3 but verify it's not 3.14+
  const fallbacks = ['/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3']
  for (const c of fallbacks) {
    try {
      const s = spawnSync(c, ['--version'], { timeout: 5000, stdio: ['ignore', 'pipe', 'pipe'] })
      if (s.status === 0) {
        const ver = s.stdout.toString().trim() // e.g. "Python 3.13.2"
        const match = ver.match(/Python 3\.(\d+)/)
        const minor = match ? parseInt(match[1], 10) : 99
        if (minor >= 10 && minor <= 13) {
          console.log(`[mlx] Found compatible Python: ${c} (${ver})`)
          return c
        } else if (minor < 10) {
          console.log(`[mlx] Skipping ${c} — ${ver} is too old (need 3.10+)`)
        } else {
          console.log(`[mlx] Skipping ${c} — ${ver} is too new for mlx-lm`)
        }
      }
    } catch {
      // not available
    }
  }

  return null
}

// ---------------------------------------------------------------------------
// MLX detection
// ---------------------------------------------------------------------------

export interface MLXStatus {
  /** Python to use for running mlx_lm (venv python if installed, system python otherwise) */
  python: string
  /** Whether mlx-lm is installed and importable */
  installed: boolean
}

/**
 * Check if mlx-lm is ready to use.
 * Returns the python path to use and whether mlx_lm is installed.
 */
export function locateMLX(): MLXStatus | null {
  // 1. Check if we have a working venv with mlx_lm installed
  const vPy = venvPython()
  if (existsSync(vPy)) {
    // Verify the venv Python is 3.10+ — older versions can't run modern mlx-lm
    try {
      const verCheck = spawnSync(vPy, ['--version'], {
        timeout: 5000,
        stdio: ['ignore', 'pipe', 'pipe']
      })
      const verStr = verCheck.stdout?.toString().trim() || ''
      const verMatch = verStr.match(/Python 3\.(\d+)/)
      const minor = verMatch ? parseInt(verMatch[1], 10) : 0
      if (minor < 10) {
        console.log(`[mlx] Existing venv uses ${verStr} (too old). Deleting and recreating…`)
        try { rmSync(venvDir(), { recursive: true, force: true }) } catch { /* ok */ }
        // Fall through to system python detection below
      } else {
        // Venv Python is compatible — check if mlx_lm is installed
        try {
          const check = spawnSync(vPy, ['-c', 'import mlx_lm; print("ok")'], {
            timeout: 15000,
            stdio: ['ignore', 'pipe', 'pipe']
          })
          const stdout = check.stdout?.toString().trim() || ''
          if (check.status === 0 && stdout.includes('ok')) {
            console.log('[mlx] Found mlx-lm in venv')
            return { python: vPy, installed: true }
          }
        } catch {
          // venv exists but mlx_lm not importable
        }
        // Venv exists but mlx_lm is missing — can still pip install into it
        return { python: vPy, installed: false }
      }
    } catch {
      // Can't check version — treat as needing recreation
      console.log('[mlx] Cannot determine venv Python version. Recreating…')
      try { rmSync(venvDir(), { recursive: true, force: true }) } catch { /* ok */ }
    }
  }

  // 2. No venv yet — find a compatible system python so we can create one
  const sysPython = findSystemPython()
  if (!sysPython) return null
  return { python: sysPython, installed: false }
}

// ---------------------------------------------------------------------------
// Installation — creates a venv and installs mlx-lm
// ---------------------------------------------------------------------------

export type InstallProgress = {
  stage: 'download' | 'install'
  message: string
}

/**
 * Install mlx-lm into a dedicated virtual environment.
 * Uses --index-url to bypass any corporate pip registries.
 * Returns the venv python path to use for all subsequent operations.
 */
export async function installMLX(
  onProgress: (p: InstallProgress) => void
): Promise<string> {
  const sysPython = findSystemPython()
  if (!sysPython) {
    throw new Error(
      'Python 3.10–3.13 not found. Please install Python via Homebrew: brew install python@3.13'
    )
  }

  const vDir = venvDir()
  const vPy = venvPython()

  // Step 1: Create venv if needed
  if (!existsSync(vPy)) {
    onProgress({ stage: 'install', message: 'Creating Python virtual environment…' })
    console.log(`[mlx] Creating venv at ${vDir} using ${sysPython}`)
    await runProcess(sysPython, ['-m', 'venv', vDir], onProgress)
  }

  // Step 2: Upgrade pip first (avoids old-pip issues)
  onProgress({ stage: 'install', message: 'Upgrading pip…' })
  await runProcess(vPy, [
    '-m', 'pip', 'install', '--upgrade', 'pip',
    '--index-url', 'https://pypi.org/simple/'
  ], onProgress)

  // Step 3: Install mlx-lm (force public PyPI to bypass corporate registries)
  onProgress({ stage: 'install', message: 'Installing mlx-lm (this may take a few minutes)…' })
  await runProcess(vPy, [
    '-m', 'pip', 'install', '--upgrade', 'mlx-lm>=0.24.0',
    '--index-url', 'https://pypi.org/simple/'
  ], onProgress)

  // Verify the install worked
  const check = spawnSync(vPy, ['-c', 'import mlx_lm; print("ok")'], {
    timeout: 15000,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  if (check.status !== 0 || !check.stdout?.toString().includes('ok')) {
    const err = check.stderr?.toString().slice(-300) || 'unknown error'
    throw new Error(`mlx-lm installed but failed to import: ${err}`)
  }

  console.log('[mlx] mlx-lm installed successfully')
  return vPy
}

/**
 * Upgrade mlx-lm to the latest PyPI release.
 * Called automatically when a CacheCompatibilityError is detected (i.e. the
 * installed mlx-lm is too old to support the model's architecture).
 */
export async function upgradeMLX(
  python: string,
  onProgress: (p: InstallProgress) => void
): Promise<void> {
  onProgress({ stage: 'install', message: 'Upgrading mlx-lm to latest version…' })
  console.log(`[mlx] Upgrading mlx-lm via ${python}`)
  await runProcess(python, [
    '-m', 'pip', 'install', '--upgrade', 'mlx-lm',
    '--index-url', 'https://pypi.org/simple/'
  ], onProgress)
  verifyMLXImport(python)
  onProgress({ stage: 'install', message: 'mlx-lm upgrade complete — restarting…' })
  console.log('[mlx] mlx-lm upgraded successfully')
}

/**
 * Install mlx-lm directly from GitHub source.
 *
 * PyPI releases can lag behind the GitHub main branch by days to weeks.
 * New model architectures (like Gemma 4's k_norm / per-layer k_proj) are
 * often supported on GitHub first. This is the escalation path when a PyPI
 * upgrade doesn't fix the incompatibility.
 *
 * Requires git to be available on PATH (standard on macOS with Xcode CLT).
 */
export async function upgradeMLXFromSource(
  python: string,
  onProgress: (p: InstallProgress) => void
): Promise<void> {
  // Use the tarball URL rather than git+https:// because:
  //  - no git dependency (not always on PATH inside the app sandbox)
  //  - downloads only the source archive, not the full git history
  //  - pip shows download progress for a plain https URL
  const tarballUrl = 'https://github.com/ml-explore/mlx-lm/archive/refs/heads/main.tar.gz'
  onProgress({ stage: 'install', message: 'Downloading mlx-lm source from GitHub…' })
  console.log(`[mlx] Installing mlx-lm from tarball ${tarballUrl} via ${python}`)
  await runProcess(python, [
    '-m', 'pip', 'install',
    '--upgrade',
    '--force-reinstall',
    tarballUrl
  ], onProgress)
  verifyMLXImport(python)
  onProgress({ stage: 'install', message: 'mlx-lm (source) installed — restarting…' })
  console.log('[mlx] mlx-lm from GitHub source installed successfully')
}

function verifyMLXImport(python: string): void {
  const check = spawnSync(python, ['-c', 'import mlx_lm; print("ok")'], {
    timeout: 15000,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  if (check.status !== 0 || !check.stdout?.toString().includes('ok')) {
    const err = check.stderr?.toString().slice(-300) || 'unknown error'
    throw new Error(`mlx-lm installed but failed to import: ${err}`)
  }
}

/** Run a subprocess and stream output to onProgress */
function runProcess(
  cmd: string,
  args: string[],
  onProgress: (p: InstallProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PIP_DISABLE_PIP_VERSION_CHECK: '1',
        // Force public PyPI — don't inherit corporate pip.conf
        PIP_INDEX_URL: 'https://pypi.org/simple/',
        PIP_EXTRA_INDEX_URL: ''
      }
    })

    let stderr = ''
    proc.stdout?.on('data', (d) => {
      const line = d.toString().trim()
      if (line) onProgress({ stage: 'install', message: line.slice(0, 120) })
    })
    proc.stderr?.on('data', (d) => {
      stderr += d.toString()
      const line = d.toString().trim()
      if (line) onProgress({ stage: 'install', message: line.slice(0, 120) })
    })
    proc.on('error', reject)
    proc.on('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`${cmd} ${args.slice(0, 3).join(' ')} failed (exit ${code}): ${stderr.slice(-500)}`))
    })
  })
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/**
 * Thrown when the cached model weights are not compatible with the current
 * version of mlx-lm (e.g. the model was updated and the cache is stale).
 * The UI shows a "Clear cache & re-download" button for this error type.
 */
export class CacheCompatibilityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CacheCompatibilityError'
  }
}

/**
 * Delete the HuggingFace Hub cache for a specific model so it can be
 * re-downloaded cleanly on the next startup.
 */
export function clearModelCache(model: string): void {
  // HuggingFace converts 'org/model' → 'models--org--model'
  const folder = 'models--' + model.replace('/', '--')
  // Try both common cache layouts (HF_HOME/hub/ and HF_HOME/ directly)
  const candidates = [
    join(modelsDir(), 'hub', folder),
    join(modelsDir(), folder)
  ]
  for (const dir of candidates) {
    if (existsSync(dir)) {
      console.log(`[mlx] Deleting stale model cache: ${dir}`)
      rmSync(dir, { recursive: true, force: true })
    }
  }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

export interface ServerProgress {
  /** Which setup stage this update belongs to */
  stage?: 'downloading-model' | 'loading-model' | 'starting-server'
  message: string
  /** 0.0–1.0 progress fraction, if available */
  progress?: number
  /** Per-file detail line (file name, transfer size, speed) */
  detail?: string
}

export async function startServer(
  python: string,
  model: string,
  onProgress?: (p: ServerProgress) => void
): Promise<void> {
  // If server is already running with the correct model, still do a health
  // check to confirm it is actually responding before declaring ready.
  if (serverProc && !serverProc.killed && currentModel === model) {
    if (onProgress) {
      onProgress({ stage: 'starting-server', message: 'Connecting to existing server…' })
    }
    try {
      const res = await fetch(`${MLX_URL}/v1/models`)
      if (res.ok) {
        console.log('[mlx] Existing server is healthy, reusing.')
        return
      }
    } catch {
      // Not responding — fall through and restart
      console.log('[mlx] Existing server not responding, restarting…')
    }
  }

  // Kill any existing server and wait for it to release the port before spawning
  await stopServer()

  const env = {
    ...process.env,
    HF_HOME: modelsDir(),
    TRANSFORMERS_CACHE: modelsDir(),
    HF_HUB_DISABLE_TELEMETRY: '1',
    // Suppress "unauthenticated requests" noise — we intentionally run without a token
    HF_HUB_DISABLE_IMPLICIT_TOKEN: '1',
    HUGGINGFACE_HUB_VERBOSITY: 'error'
  }

  // Track early exit so waitForHealth can bail out immediately
  let earlyExit: { code: number | null; stderr: string } | null = null
  let stderrBuf = ''

  // Phase flags — raised by the stderr parser, read by the health-check heartbeat
  let downloadComplete = false
  let loadingStarted = false
  let serverStarting = false
  // Track whether we have seen any HF download progress so we know if the
  // model is loading from cache (no download output) vs. downloading fresh.
  let hasSeenDownloadProgress = false
  // Track the highest layer index seen so weight-loading messages can say
  // "layer N of N" once we've seen a few layers.
  let maxLayerSeen = -1

  const spawnTime = Date.now()

  console.log(`[mlx] Starting server: ${python} -m mlx_lm server --model ${model} --port ${MLX_PORT}`)

  serverProc = spawn(
    python,
    ['-m', 'mlx_lm', 'server', '--model', model, '--port', String(MLX_PORT)],
    {
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false
    }
  )
  currentModel = model

  function handleProgress(p: ServerProgress & { _layerNum?: number }): void {
    if (!onProgress) return
    if (p.stage === 'loading-model') loadingStarted = true
    if (p.stage === 'starting-server') serverStarting = true
    if (p.stage === 'downloading-model') {
      hasSeenDownloadProgress = true
      if (p.progress === 1.0) downloadComplete = true
    }
    // Track highest layer index seen — used to annotate "layer N of N"
    if (p._layerNum !== undefined && p._layerNum > maxLayerSeen) {
      maxLayerSeen = p._layerNum
    }
    // Strip internal-only field before forwarding
    const { _layerNum: _, ...forward } = p
    onProgress(forward)
  }

  serverProc.stdout?.on('data', (d) => {
    const text = d.toString()
    console.log('[mlx stdout]', text.trim())
    if (onProgress) parseServerOutput(text, handleProgress, () => maxLayerSeen)
  })

  serverProc.stderr?.on('data', (d) => {
    const text = d.toString()
    stderrBuf += text
    console.log('[mlx stderr]', text.trim())
    if (onProgress) parseServerOutput(text, handleProgress, () => maxLayerSeen)
  })

  serverProc.on('exit', (code) => {
    console.log('[mlx] server exited with code', code)
    earlyExit = { code, stderr: stderrBuf }
    serverProc = null
    currentModel = null
  })

  // Wait for the server to become healthy.
  // First run downloads model weights from HuggingFace, so allow up to 10 min.
  await waitForHealth(600_000, () => earlyExit, onProgress ? (elapsedSec) => {
    const timeSinceSpawn = Date.now() - spawnTime

    if (serverStarting) {
      onProgress({ stage: 'starting-server', message: `Waiting for server to respond… (${elapsedSec}s)` })
      return
    }

    if (downloadComplete || loadingStarted) {
      onProgress({ stage: 'loading-model', message: `Loading model into memory… (${elapsedSec}s elapsed)` })
      return
    }

    // No download or loading messages seen yet. After a few seconds assume we
    // are loading from cache (model was already downloaded) and say so.
    if (!hasSeenDownloadProgress && timeSinceSpawn > 4000) {
      loadingStarted = true // prevent re-entering the download branch
      onProgress({ stage: 'loading-model', message: `Loading model from cache… (${elapsedSec}s elapsed)` })
      return
    }

    // Still very early — server just spawned, no output yet
    if (timeSinceSpawn <= 4000) {
      onProgress({ stage: 'loading-model', message: 'Initialising model runtime…' })
    }
  } : undefined)

  // ── Warm-up / verification ───────────────────────────────────────────────
  //
  // mlx_lm.server loads the model LAZILY — the HTTP server comes up quickly
  // and responds 200 to /v1/models before a single weight is mapped into RAM.
  // Without an explicit warm-up, the app would show the chat UI immediately,
  // then fail on the first message.
  //
  // We fire a minimal chat completion request here. This blocks until the
  // model is loaded and returns the first token (or fails with a clear error).
  // Stderr keeps flowing during this wait, so layer-loading progress still
  // updates the UI.
  const elapsed = Date.now() - spawnTime
  const verifyTimeout = Math.max(30_000, 600_000 - elapsed)
  await verifyModelReady(model, verifyTimeout, () => stderrBuf, onProgress ? handleProgress : undefined)
}

/**
 * Parse a chunk of text output from mlx_lm.server and emit structured progress.
 *
 * mlx_lm writes to stderr (and sometimes stdout). Key patterns:
 *
 *  Download phase (HuggingFace Hub tqdm bars written to stderr):
 *    "Fetching 8 files:  50%|████ | 4/8 [00:55<00:59, 14.98s/it]"
 *    "model.safetensors:  45%|████ | 1.23G/2.74G [01:23<01:40, 15.2MB/s]"
 *    "tokenizer.json: 100%|██| 428k/428k [00:01<00:00, 312kB/s]"
 *
 *  Post-download loading phase:
 *    "Fetching 8 files: 100%|████| 8/8 [04:12<00:00, 31.60s/it]"
 *    "Loading model from /path/to/cache"
 *    "Loading weights" / "Loading model weights"
 *
 *  Server-start phase:
 *    "Starting httpd at 127.0.0.1:11434"
 */
function parseServerOutput(
  text: string,
  emit: (p: ServerProgress & { _layerNum?: number }) => void,
  getMaxLayer: () => number = () => -1
): void {
  // Split on \r or \n so tqdm overwrite-style bars show as separate entries
  const lines = text.split(/[\r\n]+/)

  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue

    // ── Skip noisy server runtime lines ────────────────────────────────────
    // HTTP access log: "127.0.0.1 - - [09/May/2026 23:07:01] "POST /v1/... 200 -"
    if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} - - \[/.test(line)) continue
    // Python logging: "2026-05-09 23:07:01,977 - INFO - Prompt Cache: ..."
    if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - (INFO|WARNING|ERROR|DEBUG) -/.test(line)) continue

    // ── Server startup ──────────────────────────────────────────────────────
    if (
      /Starting httpd/i.test(line) ||
      /Server started/i.test(line) ||
      /Uvicorn running/i.test(line) ||
      /Application startup complete/i.test(line)
    ) {
      emit({ stage: 'starting-server', message: 'Server is starting up…', progress: 1.0 })
      continue
    }

    // ── Post-download model loading ─────────────────────────────────────────
    if (
      /Loading model (from|weights|checkpoint)/i.test(line) ||
      /^Loading weights/i.test(line) ||
      /Compiling model/i.test(line) ||
      /Warming up/i.test(line) ||
      /Tokenizer/i.test(line) ||
      /quantiz/i.test(line) ||
      /shard/i.test(line) ||
      /mlx\.core/i.test(line) ||
      /metal/i.test(line) ||
      /Fusing/i.test(line)
    ) {
      emit({ stage: 'loading-model', message: 'Loading model…', detail: line.slice(0, 140) })
      continue
    }

    // ── Overall HuggingFace fetch bar ───────────────────────────────────────
    // "Fetching 8 files:  50%|████ | 4/8 [00:55<00:59, 14.98s/it]"
    const fetchMatch = line.match(
      /Fetching\s+(\d+)\s+files?:\s+(\d+)%[^\d]*(\d+)\/(\d+)/
    )
    if (fetchMatch) {
      const pct = parseInt(fetchMatch[2], 10)
      const done = parseInt(fetchMatch[3], 10)
      const total = parseInt(fetchMatch[4], 10)

      if (pct === 100) {
        emit({
          stage: 'loading-model',
          message: `All ${total} file${total === 1 ? '' : 's'} downloaded — loading model into memory…`,
          progress: 1.0
        })
      } else {
        emit({
          stage: 'downloading-model',
          message: `Downloading model files — ${done} of ${total} complete (${pct}%)`,
          progress: pct / 100
        })
      }
      continue
    }

    // ── Per-file tqdm bar ───────────────────────────────────────────────────
    // "model-00001-of-00004.safetensors:  45%|█▍ | 1.23G/2.74G [01:23<01:40, 15.2MB/s]"
    // "tokenizer.json: 100%|██| 428k/428k [00:01<00:00, 312kB/s]"
    const fileMatch = line.match(
      /^([\w.\-]+\.\w+):\s+(\d+)%[^\d]*([\d.]+\s*[KMGTP]?i?[Bb]?)\/([\d.]+\s*[KMGTP]?i?[Bb]?)\s+\[([^\]]+)\]/i
    )
    if (fileMatch) {
      const filename = fileMatch[1]
      const pct = parseInt(fileMatch[2], 10)
      const done = fileMatch[3].trim()
      const total = fileMatch[4].trim()
      const timeInfo = fileMatch[5]
      const speedMatch = timeInfo.match(/([\d.]+\s*[KMGTP]?[Bi]*\/s)/i)
      const speed = speedMatch ? speedMatch[1] : ''

      if (pct === 100) {
        emit({
          stage: 'downloading-model',
          message: `Downloaded ${filename}`,
          detail: `${total} total`
        })
      } else {
        emit({
          stage: 'downloading-model',
          message: `Downloading ${filename} (${pct}%)`,
          detail: speed ? `${done} / ${total}  ·  ${speed}` : `${done} / ${total}`
        })
      }
      continue
    }

    // ── Per-layer weight tensor lines ───────────────────────────────────────
    // mlx_lm prints each weight key as it loads it, e.g.:
    //   "language_model.model.layers.34.self_attn.v_proj.weight."
    //   "model.embed_tokens.weight."
    // We parse the layer number so we can show "Loading weights — layer N of N"
    const layerMatch = line.match(/\.layers\.(\d+)\./)
    if (layerMatch) {
      const layerNum = parseInt(layerMatch[1], 10)
      const prevMax = getMaxLayer()
      // Strip trailing dot that mlx_lm sometimes appends
      const cleanKey = line.replace(/\.$/, '')
      // Extract just the component name after the layer (e.g. "self_attn.v_proj")
      const componentMatch = cleanKey.match(/\.layers\.\d+\.(.+)/)
      const component = componentMatch ? componentMatch[1] : cleanKey
      // Once we've seen multiple layers, we know the highest so far — show it
      const maxKnown = Math.max(prevMax, layerNum)
      const layerLabel = prevMax > 0
        ? `layer ${layerNum} of ${maxKnown}`
        : `layer ${layerNum}`
      emit({
        stage: 'loading-model',
        message: `Loading weights — ${layerLabel}`,
        detail: component.slice(0, 80),
        _layerNum: layerNum
      })
      continue
    }

    // ── Other weight / tensor lines (embed, norm, lm_head, etc.) ────────────
    if (/\.(weight|bias|norm|embed|proj|head)\b/.test(line) && !/[█▏▎▍▌▋▊▉▐░▒▓]/.test(line)) {
      const cleanKey = line.replace(/\.$/, '')
      emit({ stage: 'loading-model', message: 'Loading weights…', detail: cleanKey.slice(0, 120) })
      continue
    }

    // ── Generic download/fetch mention ──────────────────────────────────────
    if (/\bdownload\b|\bfetch\b/i.test(line) && line.length < 200) {
      emit({ stage: 'downloading-model', message: line.slice(0, 120) })
      continue
    }

    // ── Catch-all: forward any readable non-tqdm line as loading detail ─────
    // Avoids forwarding raw tqdm bar lines full of unicode block characters
    // but surfaces any other informational message mlx_lm emits.
    const hasBlockChars = /[█▏▎▍▌▋▊▉▐░▒▓]/.test(line)
    const looksLikePath = /\//.test(line)
    if (!hasBlockChars && line.length > 4 && line.length < 200 && (looksLikePath || /[a-zA-Z]{4,}/.test(line))) {
      emit({ stage: 'loading-model', message: 'Loading model…', detail: line.slice(0, 140) })
    }
  }
}

/**
 * Stop the server process and wait for it to fully exit before returning.
 *
 * Sending SIGTERM and returning immediately causes a race: the Python process
 * holds port 11434 for 1–3 s during cleanup, so a new spawn attempted right
 * after sees "Address already in use". Awaiting the exit event (with a 3-s
 * SIGKILL fallback) guarantees the port is free before the caller proceeds.
 */
export function stopServer(): Promise<void> {
  if (!serverProc || serverProc.killed) {
    serverProc = null
    currentModel = null
    return Promise.resolve()
  }
  const proc = serverProc
  serverProc = null
  currentModel = null
  console.log('[mlx] Stopping server — waiting for process to exit')
  return new Promise<void>((resolve) => {
    // Force-kill if SIGTERM hasn't worked within 3 s
    const killTimer = setTimeout(() => {
      console.log('[mlx] Process did not exit within 3 s — sending SIGKILL')
      try { proc.kill('SIGKILL') } catch { /* already dead */ }
      resolve()
    }, 3000)
    proc.once('exit', () => {
      clearTimeout(killTimer)
      console.log('[mlx] Server process exited cleanly')
      resolve()
    })
    proc.kill('SIGTERM')
  })
}

/**
 * Trigger the model to load by sending a minimal chat completion request,
 * then wait for the response.
 *
 * mlx_lm.server loads the model lazily on the first generation request, so
 * the /v1/models health check passes while the model is still on disk.
 * This function surfaces that load (and any incompatibility errors) before
 * the user sees the chat screen.
 *
 * Key edge-case: when mlx_lm's inference thread crashes (e.g. with
 * "ValueError: Received N parameters not in model"), it dies without ever
 * sending an HTTP response. The fetch would then block until our AbortController
 * fires — potentially for minutes. To avoid that, a stderr poller runs every
 * 2 s and aborts the fetch as soon as it detects the fatal exception signature.
 */
async function verifyModelReady(
  model: string,
  timeoutMs: number,
  getStderr: () => string,
  onProgress?: (p: ServerProgress & { _layerNum?: number }) => void
): Promise<void> {
  if (onProgress) {
    onProgress({ stage: 'loading-model', message: 'Verifying model is ready…' })
  }

  const controller = new AbortController()

  // Track whether we aborted due to a detected fatal stderr error vs. timeout
  let fatalStderrError: string | null = null

  // Poll stderr every 2 s for the Thread-1 crash signature.
  // When detected, abort the hanging fetch immediately so the caller can
  // throw a typed error within ~2 s instead of waiting for the full timeout.
  const stderrPollerId = setInterval(() => {
    const stderr = getStderr()
    const hasThreadCrash = stderr.includes('Exception in thread')
    const hasParamError =
      stderr.includes('parameters not in model') ||
      (stderr.includes('ValueError') && stderr.includes('Received') && stderr.includes('parameters'))
    if (hasThreadCrash && hasParamError) {
      fatalStderrError = stderr
      console.log('[mlx] Fatal thread crash detected in stderr — aborting warm-up fetch')
      controller.abort()
    }
  }, 2000)

  // Heartbeat interval so the elapsed timer keeps ticking while fetch blocks
  const heartbeatId = onProgress
    ? setInterval(() => {
        // The main heartbeat in waitForHealth has already stopped; keep emitting
        // so the UI doesn't freeze during the warm-up fetch.
        onProgress({ stage: 'loading-model', message: 'Verifying model is ready…' })
      }, 3000)
    : null

  try {
    const res = await fetch(`${MLX_URL}/v1/chat/completions`, {
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
      const stderr = getStderr()
      // Detect the well-known mlx-lm version-incompatibility signature
      const isCacheIssue =
        stderr.includes('parameters not in model') ||
        (stderr.includes('ValueError') && stderr.includes('Received') && stderr.includes('parameters'))
      if (isCacheIssue) {
        throw new CacheCompatibilityError(
          'The installed mlx-lm version does not support this model\'s architecture. ' +
          'The app will attempt to upgrade mlx-lm automatically.'
        )
      }
      throw new Error(
        `Model failed to load (HTTP ${res.status}). ` +
        `Server said: ${(body || 'no body').slice(0, 200)}`
      )
    }

    console.log('[mlx] Model warm-up complete — model is loaded and responding')
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      // Distinguish: did we abort because we detected a fatal error, or timed out?
      if (fatalStderrError) {
        throw new CacheCompatibilityError(
          'The installed mlx-lm version does not support this model\'s architecture. ' +
          'The app will attempt to upgrade mlx-lm automatically.'
        )
      }
      throw new Error(`Model did not finish loading within ${Math.round(timeoutMs / 1000)}s.`)
    }
    throw e
  } finally {
    controller.abort()
    clearInterval(stderrPollerId)
    if (heartbeatId) clearInterval(heartbeatId)
  }
}

/**
 * Poll the server's /v1/models endpoint until it responds.
 * Calls onHeartbeat every ~3 seconds so the UI can show elapsed-time messages
 * during the silent loading phase (weights being mapped into RAM).
 */
async function waitForHealth(
  timeoutMs: number,
  checkEarlyExit: () => { code: number | null; stderr: string } | null,
  onHeartbeat?: (elapsedSec: number) => void
): Promise<void> {
  const start = Date.now()
  let lastError: unknown = null
  let lastHeartbeat = 0

  while (Date.now() - start < timeoutMs) {
    const exit = checkEarlyExit()
    if (exit) {
      throw new Error(
        `MLX server exited with code ${exit.code}.\n\n${exit.stderr.slice(-800)}`
      )
    }

    try {
      const res = await fetch(`${MLX_URL}/v1/models`)
      if (res.ok) {
        console.log('[mlx] Server is healthy')
        return
      }
    } catch (e) {
      lastError = e
    }

    const elapsed = Math.round((Date.now() - start) / 1000)
    if (onHeartbeat && Date.now() - lastHeartbeat > 3000) {
      lastHeartbeat = Date.now()
      onHeartbeat(elapsed)
    }

    await new Promise((r) => setTimeout(r, 1500))
  }
  throw new Error(`MLX server did not respond within ${timeoutMs / 1000}s: ${String(lastError)}`)
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

export async function listLocalModels(): Promise<string[]> {
  try {
    const res = await fetch(`${MLX_URL}/v1/models`)
    if (!res.ok) return []
    const data = (await res.json()) as { data?: Array<{ id: string }> }
    return (data.data ?? []).map((m) => m.id)
  } catch {
    return []
  }
}

export async function hasModel(_name: string): Promise<boolean> {
  try {
    const models = await listLocalModels()
    return models.length > 0
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// Chat streaming (OpenAI-compatible SSE)
// ---------------------------------------------------------------------------

export interface MLXChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  images?: string[]
}

export interface MLXChatOptions {
  model: string
  messages: MLXChatMessage[]
  signal?: AbortSignal
  temperature?: number
}

export async function* chatStream(
  opts: MLXChatOptions
): AsyncGenerator<{ content?: string; done?: boolean }> {
  const res = await fetch(`${MLX_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: opts.model,
      messages: opts.messages.map((m) => ({
        role: m.role,
        content: m.content
      })),
      stream: true,
      temperature: opts.temperature ?? 0.7,
      max_tokens: 8192
    }),
    signal: opts.signal
  })

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(`Chat request failed: ${res.status} ${res.statusText} — ${text}`)
  }

  // Parse SSE stream (OpenAI format: "data: {...}\n\n")
  const stream = res.body as unknown as ReadableStream<Uint8Array>
  for await (const event of readSSE(stream)) {
    if (event === '[DONE]') {
      yield { done: true }
      return
    }
    try {
      const parsed = JSON.parse(event) as {
        choices?: Array<{
          delta?: { content?: string; role?: string }
          finish_reason?: string | null
        }>
      }
      const choice = parsed.choices?.[0]
      if (choice?.delta?.content) {
        yield { content: choice.delta.content }
      }
      if (choice?.finish_reason === 'stop' || choice?.finish_reason === 'length') {
        yield { done: true }
        return
      }
    } catch {
      // Skip malformed events
    }
  }
  yield { done: true }
}

/** Parse an SSE byte stream into individual data payloads */
async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 2)
      if (!block) continue
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data) yield data
        }
      }
    }
  }

  // Flush remaining buffer
  if (buf.trim()) {
    for (const line of buf.trim().split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (data) yield data
      }
    }
  }
}

export { MLX_URL }

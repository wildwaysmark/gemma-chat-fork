import { useEffect, useState } from 'react'
import { AVAILABLE_MODELS, type SetupStatus } from '@shared/types'
import gemmaLogoUrl from '../assets/gemma-logo.png'

interface Props {
  status: SetupStatus
  model: string
  /** Timestamp (Date.now()) when the current slow stage was entered, for elapsed timer */
  loadingStageEnteredAt?: number
  onModelChange: (m: string) => void
  onStart: (model: string) => void
}

function formatBytes(n?: number): string {
  if (!n) return ''
  const u = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let v = n
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}

/** Counts up from the given anchor timestamp while active is true. */
function useElapsedSeconds(active: boolean, anchorMs?: number): number {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!active) {
      setElapsed(0)
      return
    }
    const base = anchorMs ?? Date.now()
    const tick = (): void => setElapsed(Math.round((Date.now() - base) / 1000))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [active, anchorMs])
  return elapsed
}

export default function Setup({ status, model, loadingStageEnteredAt, onModelChange, onStart }: Props) {
  const isSlowStage = status.stage === 'loading-model' || status.stage === 'starting-server'
  const elapsed = useElapsedSeconds(isSlowStage, loadingStageEnteredAt)

  const isWorking =
    status.stage === 'checking' ||
    status.stage === 'installing-mlx' ||
    status.stage === 'starting-mlx' ||
    status.stage === 'downloading-model' ||
    status.stage === 'loading-model' ||
    status.stage === 'starting-server'

  if (status.stage === 'checking' && status.message === 'Welcome') {
    return <WelcomeScreen model={model} onModelChange={onModelChange} onStart={onStart} />
  }

  const showProgressBar =
    isWorking &&
    status.stage === 'downloading-model' &&
    status.progress != null

  // Show a long-wait warning after 90 seconds in a slow stage
  const showLongWaitHint = isSlowStage && elapsed > 90

  return (
    <div className="drag flex h-full w-full flex-col">
      <div className="h-9" />
      <div className="flex flex-1 items-center justify-center px-8">
        <div className="no-drag w-full max-w-md">
          <div className="mb-8 text-center">
            <GemmaLogo className="mx-auto mb-5 h-20 w-20" />
            <h1 className="text-[22px] font-semibold tracking-tight">Setting things up</h1>
            <p className="mt-1.5 text-sm text-ink-400">
              Everything runs locally. Nothing leaves your Mac.
            </p>
          </div>

          <StageList status={status} />

          {showProgressBar && (
            <div className="mt-6">
              <div className="h-[3px] w-full overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full bg-white/70 transition-[width] duration-200 ease-out"
                  style={{ width: `${Math.max(2, Math.round((status.progress ?? 0) * 100))}%` }}
                />
              </div>
              <div className="mt-2 flex justify-between text-[11px] tabular-nums text-ink-400">
                <span>{Math.round((status.progress ?? 0) * 100)}%</span>
                {status.bytesDone != null && status.bytesTotal != null && (
                  <span>
                    {formatBytes(status.bytesDone)} / {formatBytes(status.bytesTotal)}
                  </span>
                )}
              </div>
              {status.detail && (
                <div className="mt-1 truncate rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-ink-400">
                  {status.detail}
                </div>
              )}
            </div>
          )}

          {/* Loading / server-starting panel with live elapsed counter */}
          {isSlowStage && (
            <div className="mt-6 space-y-3">
              <div className="rounded-lg border border-white/8 bg-white/[0.03] px-4 py-3">
                {/* Header row: shimmer + message + elapsed timer */}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <div className="shimmer h-1 w-12 shrink-0 rounded-full" />
                    <span className="truncate text-[12px] text-ink-300">{status.message}</span>
                  </div>
                  <span className="shrink-0 tabular-nums text-[11px] text-ink-500">{elapsed}s</span>
                </div>

                {/* Last line of mlx output, if any */}
                {status.detail && (
                  <div className="mt-2 truncate rounded bg-black/30 px-2 py-1 font-mono text-[10px] text-ink-500">
                    {status.detail}
                  </div>
                )}

                {/* Contextual explanation */}
                <p className="mt-2 text-[11px] leading-relaxed text-ink-500">
                  {status.stage === 'loading-model'
                    ? 'Mapping model weights into RAM. Large models can take 1–2 minutes on first load.'
                    : 'Inference server is initialising — waiting for it to respond.'}
                </p>
              </div>

              {/* Long-wait hint after 90 s */}
              {showLongWaitHint && (
                <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 px-4 py-3 text-[11px] leading-relaxed text-yellow-300/70">
                  <span className="font-medium text-yellow-300/90">Still loading ({elapsed}s).</span>{' '}
                  {status.stage === 'loading-model'
                    ? 'This is normal for large models on the first run — the OS is paging weights into RAM from disk. Activity Monitor → CPU tab should show Python actively running.'
                    : 'The server process is running but not yet accepting connections. Check Activity Monitor if this continues.'}
                </div>
              )}
            </div>
          )}

          {status.stage === 'error' && (
            <div className="mt-6 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
              <div className="font-medium">
                {status.cacheError ? 'Cached model is incompatible' : 'Something went wrong'}
              </div>
              <div className="mt-1 whitespace-pre-wrap break-words text-[12px] text-red-300/80">
                {status.error}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {status.cacheError && (
                  <button
                    onClick={() => {
                      window.api.clearModelCache(model).then(() => onStart(model))
                    }}
                    className="rounded-md border border-red-400/30 bg-red-500/20 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/30"
                  >
                    Clear cache &amp; re-download
                  </button>
                )}
                <button
                  onClick={() => onStart(model)}
                  className="rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-xs hover:bg-white/10"
                >
                  Try again
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function WelcomeScreen({
  model,
  onModelChange,
  onStart
}: {
  model: string
  onModelChange: (m: string) => void
  onStart: (model: string) => void
}) {
  const selected = AVAILABLE_MODELS.find((m) => m.name === model) ?? AVAILABLE_MODELS[1]
  return (
    <div className="drag flex h-full w-full flex-col">
      <div className="h-9" />
      <div className="flex flex-1 items-center justify-center px-8">
        <div className="no-drag w-full max-w-md">
          <div className="anim-fade-up mb-8 text-center">
            <GemmaLogo className="mx-auto mb-5 h-24 w-24" />
            <h1 className="text-[26px] font-semibold tracking-tight">Welcome to Gemma Chat</h1>
            <p className="mt-2 text-[13.5px] leading-relaxed text-ink-400">
              A local AI assistant, powered by Google's Gemma 4.
              <br />
              Runs 100% on your Mac. No account, no cloud.
            </p>
          </div>

          <div className="mb-3 text-[11px] font-medium uppercase tracking-wider text-ink-400">
            Pick a model
          </div>
          <div className="anim-stagger space-y-2">
            {AVAILABLE_MODELS.map((m) => (
              <button
                key={m.name}
                onClick={() => onModelChange(m.name)}
                className={`anim-fade-up group relative w-full rounded-xl border px-4 py-3 text-left transition active:scale-[0.99] ${
                  model === m.name
                    ? 'border-white/25 bg-white/[0.06]'
                    : 'border-white/5 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{m.label}</span>
                    {m.recommended && (
                      <span className="rounded-full bg-white/10 px-2 py-[1px] text-[10px] font-medium uppercase tracking-wider text-ink-100">
                        Recommended
                      </span>
                    )}
                  </div>
                  <span className="text-xs tabular-nums text-ink-400">{m.size}</span>
                </div>
                <div className="mt-1 text-[12.5px] leading-snug text-ink-400">
                  {m.description}
                </div>
              </button>
            ))}
          </div>

          <button
            onClick={() => onStart(selected.name)}
            className="mt-6 w-full rounded-xl bg-white py-3 text-sm font-medium text-ink-900 transition hover:bg-white/90 active:scale-[0.99]"
          >
            Download {selected.label} &nbsp;·&nbsp; {selected.size}
          </button>
          <p className="mt-3 text-center text-[11px] text-ink-400">
            We'll install MLX runtime if needed. Model weights are cached locally.
          </p>
        </div>
      </div>
    </div>
  )
}

function StageList({ status }: { status: SetupStatus }) {
  // Each entry in the visual list maps to one or more internal stages
  const stages: Array<{ keys: SetupStatus['stage'][]; label: string }> = [
    { keys: ['installing-mlx'],   label: 'Install MLX runtime' },
    { keys: ['starting-mlx'],     label: 'Start runtime' },
    { keys: ['downloading-model'], label: 'Download model files' },
    { keys: ['loading-model'],    label: 'Load model into memory' },
    { keys: ['starting-server'],  label: 'Start server' },
    { keys: ['ready'],            label: 'Ready to chat' },
  ]

  // Ordered list used to compute done / active / pending
  const order: SetupStatus['stage'][] = [
    'checking',
    'installing-mlx',
    'starting-mlx',
    'downloading-model',
    'loading-model',
    'starting-server',
    'ready',
  ]
  const currentIdx = order.indexOf(status.stage)

  return (
    <div className="space-y-3">
      {stages.map((s) => {
        const indices = s.keys.map((k) => order.indexOf(k))
        const minIdx = Math.min(...indices)
        const maxIdx = Math.max(...indices)

        let state: 'done' | 'active' | 'pending'
        if (maxIdx < currentIdx) state = 'done'
        else if (minIdx <= currentIdx) state = 'active'
        else state = 'pending'

        const isActive = state === 'active'

        return (
          <div key={s.keys[0]} className="flex items-start gap-3">
            <div className="mt-[2px]">
              <StageDot state={state} />
            </div>
            <div className="flex-1 min-w-0">
              <div
                className={`text-sm transition ${
                  state === 'pending'
                    ? 'text-ink-400'
                    : state === 'active'
                      ? 'text-white'
                      : 'text-ink-200'
                }`}
              >
                {isActive && status.message ? status.message : s.label}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function StageDot({ state }: { state: 'pending' | 'active' | 'done' }) {
  if (state === 'done') {
    return (
      <div className="flex h-5 w-5 items-center justify-center rounded-full bg-white/85 text-ink-900">
        <svg viewBox="0 0 12 12" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M2.5 6.5l2.5 2.5 4.5-5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    )
  }
  if (state === 'active') {
    return (
      <div className="relative flex h-5 w-5 items-center justify-center">
        <div className="absolute inset-0 animate-ping rounded-full bg-white/30" />
        <div className="h-2 w-2 rounded-full bg-white" />
      </div>
    )
  }
  return <div className="h-5 w-5 rounded-full border border-white/15" />
}

function GemmaLogo({ className }: { className?: string }) {
  return (
    <img
      src={gemmaLogoUrl}
      alt="Gemma"
      className={className}
      draggable={false}
    />
  )
}

import { useEffect, useRef, useState } from 'react'
import { MLX_MODELS, OLLAMA_MODELS, type SetupStatus, type ModelInfo } from '@shared/types'
import Setup from './components/Setup'
import Chat from './components/Chat'

const PLATFORM = window.api.platform
const DEFAULT_MODEL = window.api.defaultModel
const AVAILABLE_MODELS: ModelInfo[] = PLATFORM === 'win32' ? OLLAMA_MODELS : MLX_MODELS

type AppState =
  | { phase: 'boot' }
  | { phase: 'setup'; status: SetupStatus; model: string; loadingStageEnteredAt?: number }
  | { phase: 'ready'; model: string }
  | { phase: 'switching'; model: string; toModel: string; status: SetupStatus; loadingStageEnteredAt?: number }

export default function App() {
  const [state, setState] = useState<AppState>({ phase: 'boot' })
  // Track the last stage so we know when loadingStageEnteredAt should reset
  const lastStageRef = useRef<string | null>(null)

  useEffect(() => {
    // Forward raw Gemma output to devtools console for debugging
    const rawUnsub = window.api.onRawChunk((ev) => {
      // eslint-disable-next-line no-console
      console.log('[gemma]', ev.chunk)
    })
    let unsub: (() => void) | undefined
    ;(async () => {
      unsub = window.api.onSetupStatus((status) => {
        setState((prev) => {
          // Once in the chat screen, ignore all setup status updates.
          // Stale stderr lines (HTTP access logs, Prompt Cache INFO messages)
          // can arrive AFTER the ready signal and would otherwise revert the
          // UI back to the setup screen via the catch-all branch below.
          if (prev.phase === 'ready') return prev

          if (status.stage === 'ready') {
            // Only transition to the chat screen from a proper setup phase.
            // If we are still in 'boot' (async init not yet complete), ignore
            // the early ready signal — the init flow will call startSetup again.
            if (prev.phase === 'switching') {
              return { phase: 'ready', model: prev.toModel }
            }
            if (prev.phase === 'setup') {
              return { phase: 'ready', model: prev.model }
            }
            // boot or already-ready: ignore to prevent premature chat display
            return prev
          }
          if (status.stage === 'error') {
            // If switch failed, revert to the previous model's chat screen
            if (prev.phase === 'switching') {
              return { phase: 'ready', model: prev.model }
            }
          }
          // Track when we enter a slow phase so the UI can show a live timer
          const isSlowStage =
            status.stage === 'installing-mlx' ||
            status.stage === 'loading-model' ||
            status.stage === 'starting-server'
          const stageChanged = lastStageRef.current !== status.stage
          if (stageChanged) lastStageRef.current = status.stage
          const loadingStageEnteredAt = isSlowStage && stageChanged
            ? Date.now()
            : (prev.phase === 'setup' || prev.phase === 'switching')
              ? prev.loadingStageEnteredAt
              : undefined

          // Keep switching phase alive while model is loading
          if (prev.phase === 'switching') {
            return { ...prev, status, loadingStageEnteredAt }
          }
          const model = prev.phase === 'setup' ? prev.model : DEFAULT_MODEL
          return { phase: 'setup', status, model, loadingStageEnteredAt }
        })
      })

      const local = await window.api.listLocalModels()
      const hasDefault = local.some(
        (m) => m === DEFAULT_MODEL || m.startsWith(DEFAULT_MODEL + ':')
      )
      if (hasDefault) {
        const { hasMLX } = await window.api.checkMLX()
        if (hasMLX) {
          setState({
            phase: 'setup',
            status: { stage: 'starting-mlx', message: 'Starting model runtime…' },
            model: DEFAULT_MODEL
          })
          window.api.startSetup(DEFAULT_MODEL)
          return
        }
      }
      setState({
        phase: 'setup',
        status: { stage: 'checking', message: 'Welcome' },
        model: DEFAULT_MODEL
      })
    })()
    return () => {
      unsub?.()
      rawUnsub?.()
    }
  }, [])

  function handleSwitchModel(newModel: string): void {
    setState((prev) => {
      if (prev.phase !== 'ready') return prev
      if (prev.model === newModel) return prev
      return {
        phase: 'switching',
        model: prev.model,
        toModel: newModel,
        status: { stage: 'downloading-model', message: 'Switching model…' }
      }
    })
    window.api.switchModel(newModel)
  }

  if (state.phase === 'boot') {
    return <BootSplash />
  }

  if (state.phase === 'setup') {
    return (
      <div key="setup" className="anim-fade-in h-full w-full">
        <Setup
          status={state.status}
          model={state.model}
          availableModels={AVAILABLE_MODELS}
          platform={PLATFORM}
          loadingStageEnteredAt={state.loadingStageEnteredAt}
          onModelChange={(m) =>
            setState((s) => (s.phase === 'setup' ? { ...s, model: m } : s))
          }
          onStart={(model) => {
            lastStageRef.current = null
            setState({
              phase: 'setup',
              status: { stage: 'checking', message: 'Checking system…' },
              model
            })
            window.api.startSetup(model)
          }}
        />
      </div>
    )
  }

  if (state.phase === 'switching') {
    return (
      <div key="switching" className="anim-fade-in h-full w-full">
        <Chat model={state.model} onSwitchModel={handleSwitchModel} />
        <SwitchingOverlay status={state.status} loadingStageEnteredAt={state.loadingStageEnteredAt} />
      </div>
    )
  }

  return (
    <div key="chat" className="anim-fade-scale h-full w-full">
      <Chat model={state.model} onSwitchModel={handleSwitchModel} />
    </div>
  )
}

function BootSplash() {
  return (
    <div className="drag flex h-full w-full items-center justify-center">
      <div className="shimmer h-1 w-40 rounded-full" />
    </div>
  )
}

function SwitchingOverlay({
  status,
  loadingStageEnteredAt
}: {
  status: SetupStatus
  loadingStageEnteredAt?: number
}) {
  const elapsed = useElapsedSeconds(
    (status.stage === 'loading-model' || status.stage === 'starting-server') && !!loadingStageEnteredAt,
    loadingStageEnteredAt
  )
  const isLoading = status.stage === 'loading-model' || status.stage === 'starting-server'
  const showBar = status.stage === 'downloading-model' && status.progress != null && status.progress > 0

  return (
    <div className="anim-fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="anim-fade-up flex w-80 flex-col items-center gap-4 rounded-2xl border border-white/10 bg-ink-950 px-8 py-8 shadow-2xl">
        <div className="shimmer h-1 w-32 rounded-full" />
        <p className="text-center text-sm text-ink-200">{status.message}</p>
        {showBar && (
          <div className="w-full">
            <div className="h-1 w-full rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-white/60 transition-all duration-500"
                style={{ width: `${Math.round((status.progress ?? 0) * 100)}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between text-[10px] tabular-nums text-ink-400">
              <span>{Math.round((status.progress ?? 0) * 100)}%</span>
              {status.detail && <span className="truncate pl-2">{status.detail}</span>}
            </div>
          </div>
        )}
        {isLoading && (
          <div className="w-full space-y-2">
            <div className="flex items-center justify-between text-[11px] text-ink-400">
              <span>{status.stage === 'loading-model' ? 'Loading weights into RAM' : 'Starting server'}</span>
              <span className="tabular-nums">{elapsed}s</span>
            </div>
            {status.detail && (
              <p className="truncate rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-ink-400">
                {status.detail}
              </p>
            )}
            <p className="text-center text-[11px] leading-relaxed text-ink-500">
              {status.stage === 'loading-model'
                ? 'Mapping model weights into RAM — may take 1–2 min for large models.'
                : 'Waiting for the inference server to respond…'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/** Counts up from zero in whole seconds while active is true. Resets on anchor change. */
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

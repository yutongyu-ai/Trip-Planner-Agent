import { useState } from 'react'
import { TripPlanError, fetchTripPlan } from './api'
import { LanguageToggle } from './components/LanguageToggle'
import { TripPlanView } from './components/TripPlanView'
import { TripSetupForm } from './components/TripSetupForm'
import { useLanguage } from './i18n'
import type { TripPlanResponse, TripSetupRequest } from './types'
import './App.css'

type ViewState =
  | { status: 'setup' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'plan'; plan: TripPlanResponse; dayStartTime: string }

function App() {
  const [state, setState] = useState<ViewState>({ status: 'setup' })
  const { lang, t } = useLanguage()

  async function handleSubmit(request: TripSetupRequest) {
    setState({ status: 'loading' })
    try {
      const plan = await fetchTripPlan(request, lang)
      setState({ status: 'plan', plan, dayStartTime: request.day_start_time })
    } catch (error) {
      const message = error instanceof TripPlanError ? error.message : t('genericError')
      setState({ status: 'error', message })
    }
  }

  function handleReset() {
    setState({ status: 'setup' })
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <LanguageToggle />
        <span className="app-kicker">{t('kicker')}</span>
        <h1>{t('title')}</h1>
        <p>{t('subtitle')}</p>
      </header>

      {state.status === 'setup' && <TripSetupForm onSubmit={handleSubmit} submitting={false} />}

      {state.status === 'loading' && (
        <div className="status-panel">
          <span className="spinner" aria-hidden="true" />
          <p>{t('loading')}</p>
        </div>
      )}

      {state.status === 'error' && (
        <div className="status-panel status-panel-error">
          <span className="status-icon" aria-hidden="true">
            !
          </span>
          <p>{state.message}</p>
          <button type="button" className="secondary-button" onClick={handleReset}>
            {t('backToSetup')}
          </button>
        </div>
      )}

      {state.status === 'plan' && (
        <TripPlanView plan={state.plan} dayStartTime={state.dayStartTime} onReset={handleReset} />
      )}
    </main>
  )
}

export default App

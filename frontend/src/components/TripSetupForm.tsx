import { useState } from 'react'
import type { FormEvent } from 'react'
import { useLanguage } from '../i18n'
import type { BudgetLevel, DailyPace, TripSetupRequest } from '../types'

interface TripSetupFormProps {
  onSubmit: (request: TripSetupRequest) => void
  submitting: boolean
}

function today(): string {
  return new Date().toISOString().slice(0, 10)
}

function nowTime(): string {
  const now = new Date()
  return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
}

export function TripSetupForm({ onSubmit, submitting }: TripSetupFormProps) {
  const { lang, t } = useLanguage()
  const [destination, setDestination] = useState('')
  const [startDate, setStartDate] = useState(today())
  const [endDate, setEndDate] = useState(today())
  const [interestsText, setInterestsText] = useState('')
  const [dailyPace, setDailyPace] = useState<DailyPace>('moderate')
  const [budgetLevel, setBudgetLevel] = useState<BudgetLevel>('moderate')
  const [dayStartTime, setDayStartTime] = useState('09:00')
  const [dayEndTime, setDayEndTime] = useState('18:00')
  const [formError, setFormError] = useState<string | null>(null)

  function handleSubmit(event: FormEvent) {
    event.preventDefault()

    if (!destination.trim()) {
      setFormError(t('errorDestinationRequired'))
      return
    }
    if (endDate < startDate) {
      setFormError(t('errorEndBeforeStart'))
      return
    }
    if (dayEndTime <= dayStartTime) {
      setFormError(t('errorDayEndBeforeStart'))
      return
    }
    if (startDate === today() && dayStartTime < nowTime()) {
      setFormError(t('errorDayStartInPast'))
      return
    }

    setFormError(null)
    onSubmit({
      destination: destination.trim(),
      start_date: startDate,
      end_date: endDate,
      interests: interestsText
        .split(',')
        .map((interest) => interest.trim())
        .filter(Boolean),
      daily_pace: dailyPace,
      budget_level: budgetLevel,
      day_start_time: dayStartTime,
      day_end_time: dayEndTime,
      lang,
    })
  }

  return (
    <form className="trip-setup-form" onSubmit={handleSubmit}>
      <label className="field">
        <span>{t('fieldDestination')}</span>
        <input
          type="text"
          value={destination}
          onChange={(event) => setDestination(event.target.value)}
          placeholder={t('destinationPlaceholder')}
          disabled={submitting}
          required
        />
      </label>

      <p className="form-section-label">{t('sectionWhen')}</p>

      <div className="field-row">
        <label className="field">
          <span>{t('fieldStartDate')}</span>
          <input
            type="date"
            value={startDate}
            min={today()}
            onChange={(event) => {
              setStartDate(event.target.value)
              if (endDate < event.target.value) setEndDate(event.target.value)
            }}
            disabled={submitting}
            required
          />
        </label>
        <label className="field">
          <span>{t('fieldEndDate')}</span>
          <input
            type="date"
            value={endDate}
            min={startDate}
            onChange={(event) => setEndDate(event.target.value)}
            disabled={submitting}
            required
          />
        </label>
      </div>

      <div className="field-row">
        <label className="field">
          <span>{t('fieldDayStart')}</span>
          <input
            type="time"
            value={dayStartTime}
            min={startDate === today() ? nowTime() : undefined}
            onChange={(event) => setDayStartTime(event.target.value)}
            disabled={submitting}
            required
          />
        </label>
        <label className="field">
          <span>{t('fieldDayEnd')}</span>
          <input
            type="time"
            value={dayEndTime}
            onChange={(event) => setDayEndTime(event.target.value)}
            disabled={submitting}
            required
          />
        </label>
      </div>

      <label className="field">
        <span>{t('fieldInterests')}</span>
        <input
          type="text"
          value={interestsText}
          onChange={(event) => setInterestsText(event.target.value)}
          placeholder={t('interestsPlaceholder')}
          disabled={submitting}
        />
        <small>{t('interestsHint')}</small>
      </label>

      <p className="form-section-label">{t('sectionPreferences')}</p>

      <div className="field-row">
        <label className="field">
          <span>{t('fieldPace')}</span>
          <select
            value={dailyPace}
            onChange={(event) => setDailyPace(event.target.value as DailyPace)}
            disabled={submitting}
          >
            <option value="relaxed">{t('paceRelaxed')}</option>
            <option value="moderate">{t('paceModerate')}</option>
            <option value="packed">{t('pacePacked')}</option>
          </select>
        </label>
        <label className="field">
          <span>{t('fieldBudget')}</span>
          <select
            value={budgetLevel}
            onChange={(event) => setBudgetLevel(event.target.value as BudgetLevel)}
            disabled={submitting}
          >
            <option value="budget">{t('budgetBudget')}</option>
            <option value="moderate">{t('budgetModerate')}</option>
            <option value="luxury">{t('budgetLuxury')}</option>
          </select>
        </label>
      </div>

      {formError && <p className="form-error">{formError}</p>}

      <button type="submit" className="primary-button" disabled={submitting}>
        {submitting ? t('submitting') : t('submit')}
      </button>
    </form>
  )
}

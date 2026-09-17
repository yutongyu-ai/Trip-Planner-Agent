import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { validateDestination } from '../api'
import { useLanguage } from '../i18n'
import type { TranslationKey } from '../i18n'
import type { BudgetLevel, DailyPace, TripSetupRequest } from '../types'

interface TripSetupFormProps {
  onSubmit: (request: TripSetupRequest) => void
  submitting: boolean
  // Repopulates the form after a failed submit that bounced the traveller
  // back here (see App.tsx) -- without this, retrying after e.g. a
  // destination_not_found response would mean retyping everything.
  initialValues?: TripSetupRequest
  // A server-side error tied to a specific field (currently only
  // destination_not_found, see App.tsx/main.py's error_code), surfaced
  // inline the same way a client-side validation error is.
  destinationError?: string
}

// Fixed vocabulary for the interest checkboxes -- `value` is what's actually
// sent as an interest string (folded into the Places search query, see
// backend/app/tools/places_client.py's _build_query), `labelKey` is only the
// displayed translation. Keeping the value list closed (rather than free
// text) means every option here has already been checked to recall well
// against Places, in either UI language.
const INTEREST_OPTIONS: { value: string; labelKey: TranslationKey }[] = [
  { value: 'history and culture', labelKey: 'interestHistoryCulture' },
  { value: 'art and museums', labelKey: 'interestArtMuseums' },
  { value: 'food and dining', labelKey: 'interestFoodDining' },
  { value: 'shopping', labelKey: 'interestShopping' },
  { value: 'nature and parks', labelKey: 'interestNatureParks' },
  { value: 'nightlife', labelKey: 'interestNightlife' },
  { value: 'family and kids', labelKey: 'interestFamilyKids' },
]

interface FieldErrors {
  destination?: string
  interests?: string
  endDate?: string
  dayStartTime?: string
  dayEndTime?: string
}

// Backed by a real limit, not just a UI nicety: candidate_pool_target (see
// backend/app/graph/nodes/_shared.py) caps the ranked candidate pool at 40,
// and MAX_TRIP_DAYS=5 (backend/app/main.py) is the longest trip that stays
// under that cap at every daily_pace, "packed" (the hungriest, 6 stops/day)
// included -- 5*6+6=36. A 6th packed day would need 42 and come back with
// unplanned days. The backend rejects a longer request outright; this is
// just so the traveller sees the limit before submitting, not after.
const MAX_TRIP_DAYS = 5

// `toISOString()` always renders in UTC, not local time -- for any timezone
// ahead of UTC (e.g. UTC+8), local midnight is still the *previous* UTC day,
// so slicing its ISO string silently returns yesterday's date. Every
// local-calendar-date helper below reads back its answer through the local
// Date getters (getFullYear/getMonth/getDate) instead, never toISOString,
// so the returned string always matches the browser's own local date.
function toLocalIsoDate(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function today(): string {
  return toLocalIsoDate(new Date())
}

function nowTime(): string {
  const now = new Date()
  return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
}

function addDays(isoDate: string, days: number): string {
  const date = new Date(`${isoDate}T00:00:00`)
  date.setDate(date.getDate() + days)
  return toLocalIsoDate(date)
}

function tripDayCount(startDate: string, endDate: string): number {
  const start = new Date(`${startDate}T00:00:00`)
  const end = new Date(`${endDate}T00:00:00`)
  return Math.round((end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000)) + 1
}

export function TripSetupForm({ onSubmit, submitting, initialValues, destinationError }: TripSetupFormProps) {
  const { lang, t } = useLanguage()
  const [destination, setDestination] = useState(initialValues?.destination ?? '')
  const [startDate, setStartDate] = useState(initialValues?.start_date ?? today())
  const [endDate, setEndDate] = useState(initialValues?.end_date ?? today())
  const [selectedInterests, setSelectedInterests] = useState<string[]>(initialValues?.interests ?? [])
  const [surpriseMe, setSurpriseMe] = useState(initialValues ? initialValues.interests.length === 0 : false)
  const [preferenceNotes, setPreferenceNotes] = useState(initialValues?.preference_notes ?? '')
  const [dailyPace, setDailyPace] = useState<DailyPace>(initialValues?.daily_pace ?? 'moderate')
  const [budgetLevel, setBudgetLevel] = useState<BudgetLevel>(initialValues?.budget_level ?? 'moderate')
  const [dayStartTime, setDayStartTime] = useState(initialValues?.day_start_time ?? '09:00')
  const [dayEndTime, setDayEndTime] = useState(initialValues?.day_end_time ?? '18:00')
  const [avoidOutdoorInBadWeather, setAvoidOutdoorInBadWeather] = useState(
    initialValues?.avoid_outdoor_in_bad_weather ?? false
  )
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [checkingDestination, setCheckingDestination] = useState(false)
  // Caches the last checked value's result so blur and submit share one
  // network call instead of two, and so submit can skip re-checking a value
  // blur already confirmed.
  const destinationCheckRef = useRef<{ value: string; message: string | null } | null>(null)

  // Surface a server-side error inline under its field, same as a
  // client-side one -- only re-runs when the prop itself changes, so it
  // never fights with the traveller clearing it by editing the field below.
  useEffect(() => {
    if (destinationError) setFieldErrors((prev) => ({ ...prev, destination: destinationError }))
  }, [destinationError])

  function clearFieldError(field: keyof FieldErrors) {
    setFieldErrors((prev) => (prev[field] ? { ...prev, [field]: undefined } : prev))
  }

  async function checkDestination(value: string): Promise<string | null> {
    setCheckingDestination(true)
    const message = await validateDestination(value, lang)
    setCheckingDestination(false)
    destinationCheckRef.current = { value, message }
    return message
  }

  // Fast early check on blur, so a bad destination is flagged before the
  // traveller fills out the rest of the form and waits through the full
  // planning pipeline. This alone can't be the only gate though -- hitting
  // Enter/Submit right after typing races this async check, so handleSubmit
  // below re-checks (or reuses this result) before ever calling onSubmit.
  async function handleDestinationBlur() {
    const value = destination.trim()
    if (!value || destinationCheckRef.current?.value === value) return
    const message = await checkDestination(value)
    // Only apply if the field still holds the value just checked -- avoids
    // a slow response landing after the traveller already changed it.
    if (message && destination.trim() === value) {
      setFieldErrors((prev) => ({ ...prev, destination: message }))
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()

    const errors: FieldErrors = {}
    if (!destination.trim()) errors.destination = t('errorDestinationRequired')
    if (!surpriseMe && selectedInterests.length === 0) errors.interests = t('errorInterestsRequired')
    if (endDate < startDate) errors.endDate = t('errorEndBeforeStart')
    else if (tripDayCount(startDate, endDate) > MAX_TRIP_DAYS) errors.endDate = t('errorTripTooLong')
    if (dayEndTime <= dayStartTime) errors.dayEndTime = t('errorDayEndBeforeStart')
    if (startDate === today() && dayStartTime < nowTime()) errors.dayStartTime = t('errorDayStartInPast')

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors)
      return
    }

    // Block on the destination check here too (not just on blur) -- this is
    // what actually keeps a bad destination from ever reaching the full
    // planning pipeline, regardless of how the traveller triggered submit.
    const value = destination.trim()
    const cached = destinationCheckRef.current
    const destinationMessage = cached?.value === value ? cached.message : await checkDestination(value)
    if (destinationMessage) {
      setFieldErrors((prev) => ({ ...prev, destination: destinationMessage }))
      return
    }

    setFieldErrors({})
    onSubmit({
      destination: destination.trim(),
      start_date: startDate,
      end_date: endDate,
      interests: surpriseMe ? [] : selectedInterests,
      ...(preferenceNotes.trim() ? { preference_notes: preferenceNotes.trim() } : {}),
      daily_pace: dailyPace,
      budget_level: budgetLevel,
      day_start_time: dayStartTime,
      day_end_time: dayEndTime,
      lang,
      avoid_outdoor_in_bad_weather: avoidOutdoorInBadWeather,
    })
  }

  return (
    <form className="trip-setup-form" onSubmit={handleSubmit} noValidate>
      <label className="field">
        <span>{t('fieldDestination')}</span>
        <input
          type="text"
          value={destination}
          onChange={(event) => {
            setDestination(event.target.value)
            clearFieldError('destination')
          }}
          onBlur={handleDestinationBlur}
          placeholder={t('destinationPlaceholder')}
          disabled={submitting}
          required
          className={fieldErrors.destination ? 'field-invalid' : undefined}
        />
        {fieldErrors.destination ? (
          <small className="field-error">{fieldErrors.destination}</small>
        ) : (
          checkingDestination && <small>{t('checkingDestination')}</small>
        )}
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
              const newStart = event.target.value
              setStartDate(newStart)
              if (endDate < newStart) setEndDate(newStart)
              else if (tripDayCount(newStart, endDate) > MAX_TRIP_DAYS) {
                setEndDate(addDays(newStart, MAX_TRIP_DAYS - 1))
              }
              clearFieldError('endDate')
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
            max={addDays(startDate, MAX_TRIP_DAYS - 1)}
            onChange={(event) => {
              setEndDate(event.target.value)
              clearFieldError('endDate')
            }}
            disabled={submitting}
            required
            className={fieldErrors.endDate ? 'field-invalid' : undefined}
          />
          {fieldErrors.endDate ? (
            <small className="field-error">{fieldErrors.endDate}</small>
          ) : (
            <small>{t('tripLengthHint')}</small>
          )}
        </label>
      </div>

      <div className="field-row">
        <label className="field">
          <span>{t('fieldDayStart')}</span>
          <input
            type="time"
            value={dayStartTime}
            min={startDate === today() ? nowTime() : undefined}
            onChange={(event) => {
              setDayStartTime(event.target.value)
              clearFieldError('dayStartTime')
            }}
            disabled={submitting}
            required
            className={fieldErrors.dayStartTime ? 'field-invalid' : undefined}
          />
          {fieldErrors.dayStartTime && <small className="field-error">{fieldErrors.dayStartTime}</small>}
        </label>
        <label className="field">
          <span>{t('fieldDayEnd')}</span>
          <input
            type="time"
            value={dayEndTime}
            onChange={(event) => {
              setDayEndTime(event.target.value)
              clearFieldError('dayEndTime')
            }}
            disabled={submitting}
            required
            className={fieldErrors.dayEndTime ? 'field-invalid' : undefined}
          />
          {fieldErrors.dayEndTime && <small className="field-error">{fieldErrors.dayEndTime}</small>}
        </label>
      </div>

      <div className="field">
        <span>{t('fieldInterests')}</span>
        <div className="interest-options">
          {INTEREST_OPTIONS.map((option) => (
            <label key={option.value} className="field-checkbox interest-option">
              <input
                type="checkbox"
                checked={selectedInterests.includes(option.value)}
                onChange={(event) => {
                  setSurpriseMe(false)
                  setSelectedInterests((prev) =>
                    event.target.checked
                      ? [...prev, option.value]
                      : prev.filter((value) => value !== option.value)
                  )
                  clearFieldError('interests')
                }}
                disabled={submitting}
              />
              <span>{t(option.labelKey)}</span>
            </label>
          ))}
          <label className="field-checkbox interest-option">
            <input
              type="checkbox"
              checked={surpriseMe}
              onChange={(event) => {
                setSurpriseMe(event.target.checked)
                if (event.target.checked) setSelectedInterests([])
                clearFieldError('interests')
              }}
              disabled={submitting}
            />
            <span>{t('interestSurpriseMe')}</span>
          </label>
        </div>
        {fieldErrors.interests ? (
          <small className="field-error">{fieldErrors.interests}</small>
        ) : (
          <small>{t('interestsHint')}</small>
        )}
      </div>

      <label className="field">
        <span>{t('fieldPreferenceNotes')}</span>
        <input
          type="text"
          value={preferenceNotes}
          onChange={(event) => setPreferenceNotes(event.target.value)}
          placeholder={t('preferenceNotesPlaceholder')}
          disabled={submitting}
        />
        <small>{t('preferenceNotesHint')}</small>
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

      <label className="field field-checkbox">
        <input
          type="checkbox"
          checked={avoidOutdoorInBadWeather}
          onChange={(event) => setAvoidOutdoorInBadWeather(event.target.checked)}
          disabled={submitting}
        />
        <span>{t('fieldAvoidOutdoorInBadWeather')}</span>
      </label>

      <button type="submit" className="primary-button" disabled={submitting || checkingDestination}>
        {submitting ? t('submitting') : checkingDestination ? t('checkingDestination') : t('submit')}
      </button>
    </form>
  )
}

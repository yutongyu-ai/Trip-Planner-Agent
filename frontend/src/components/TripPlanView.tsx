import { useLanguage } from '../i18n'
import type { Lang } from '../i18n'
import type { ItineraryDay, TripPlanResponse } from '../types'
import { ChatEditor } from './ChatEditor'

interface TripPlanViewProps {
  plan: TripPlanResponse
  dayStartTime: string // HH:MM, the day-start the itinerary's offsets are relative to
  onItineraryUpdate: (itinerary: ItineraryDay[]) => void
  onReset: () => void
}

function parseDayStartMinutes(dayStartTime: string): number {
  const [hours, minutes] = dayStartTime.split(':').map(Number)
  return hours * 60 + minutes
}

function formatArrivalTime(offsetMin: number, dayStartMinutes: number): string {
  const totalMinutes = dayStartMinutes + offsetMin
  const hours24 = Math.floor(totalMinutes / 60) % 24
  const minutes = totalMinutes % 60
  const period = hours24 >= 12 ? 'PM' : 'AM'
  const hours12 = ((hours24 + 11) % 12) + 1
  return `${hours12}:${minutes.toString().padStart(2, '0')} ${period}`
}

function formatDuration(min: number): string {
  if (min < 60) return `${min} min`
  const h = Math.floor(min / 60)
  const m = min % 60
  return m === 0 ? `${h} h` : `${h} h ${m} min`
}

const DATE_LOCALES: Record<Lang, string> = { en: 'en-US', zh: 'zh-CN' }

function formatDate(isoDate: string, lang: Lang): string {
  // Locale is driven by the app's own language toggle, not the browser's
  // (`undefined` would silently follow the visitor's OS/browser language
  // instead, which is the bug that put unwanted Chinese dates in an
  // English-only UI in the first place -- this is deliberate, not that).
  return new Date(`${isoDate}T00:00:00`).toLocaleDateString(DATE_LOCALES[lang], {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  })
}

function DayCard({ day, index, dayStartMinutes }: { day: ItineraryDay; index: number; dayStartMinutes: number }) {
  const { lang, t } = useLanguage()

  return (
    <li className="day-card">
      <div className="day-card-header">
        <h3>
          <span className="day-card-number">
            {t('dayPrefix')}
            {index + 1}
            {t('daySuffix')}
          </span>
          <span className="day-card-date">{formatDate(day.date, lang)}</span>
        </h3>
        {day.weather_note && <span className="weather-note">{day.weather_note}</span>}
      </div>
      {day.stops.length === 0 ? (
        <p className="day-empty">{t('noStops')}</p>
      ) : (
        <ol className="stop-list">
          {day.stops.map((stop, i) => {
            const prev = i > 0 ? day.stops[i - 1] : null

            if (stop.kind === 'meal') {
              const start = formatArrivalTime(stop.arrival_offset_min, dayStartMinutes)
              const end = formatArrivalTime(stop.depart_offset_min, dayStartMinutes)
              return (
                <li key={stop.place_id} className="stop-meal">
                  <span className="stop-meal-title">
                    🍽 {stop.attraction_name} · {start} – {end} ({formatDuration(stop.dwell_minutes)})
                  </span>
                  {prev && (
                    <span className="stop-meal-hint">
                      {t('mealNear').replace('{place}', prev.attraction_name)}
                    </span>
                  )}
                </li>
              )
            }

            const openClock = formatArrivalTime(
              stop.arrival_offset_min, // arrival == opening time when there's an opening wait
              dayStartMinutes,
            )
            const showTransit = prev !== null && stop.travel_from_prev_min > 0
            const showWait = stop.wait_for_opening_min > 0
            return (
              <li key={stop.place_id} className="stop">
                {(showTransit || showWait) && (
                  <span className="stop-transit">
                    {showTransit && `↓ ${t('transit')} ~${formatDuration(stop.travel_from_prev_min)}`}
                    {showWait &&
                      `${showTransit ? ' · ' : ''}${t('opensAt')} ${openClock}, ${t('wait')} ${formatDuration(stop.wait_for_opening_min)}`}
                  </span>
                )}
                <span className="stop-row">
                  <span className="stop-marker" aria-hidden="true" />
                  <span className="stop-time">{formatArrivalTime(stop.arrival_offset_min, dayStartMinutes)}</span>
                  <span className="stop-name">{stop.attraction_name}</span>
                </span>
                <span className="stop-meta">
                  {t('visitDuration')} {formatDuration(stop.dwell_minutes)} · →{' '}
                  {formatArrivalTime(stop.depart_offset_min, dayStartMinutes)}
                </span>
              </li>
            )
          })}
        </ol>
      )}
    </li>
  )
}

export function TripPlanView({ plan, dayStartTime, onItineraryUpdate, onReset }: TripPlanViewProps) {
  const { lang, t } = useLanguage()
  const dayStartMinutes = parseDayStartMinutes(dayStartTime)

  return (
    <section className="trip-plan-view">
      {lang !== 'en' && <p className="trip-results-note">{t('resultsNote')}</p>}
      {plan.notes.length > 0 && (
        <div className="trip-notes">
          <ul>
            {plan.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
      <p className="trip-summary">{plan.summary}</p>
      {plan.errors.length > 0 && (
        <div className="trip-errors">
          <ul>
            {plan.errors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        </div>
      )}
      <ol className="day-list">
        {plan.itinerary.map((day, index) => (
          <DayCard key={day.date} day={day} index={index} dayStartMinutes={dayStartMinutes} />
        ))}
      </ol>
      <ChatEditor sessionId={plan.session_id} onItineraryUpdate={onItineraryUpdate} />
      <button type="button" className="secondary-button" onClick={onReset}>
        {t('planAnother')}
      </button>
    </section>
  )
}

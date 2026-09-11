export type DailyPace = 'relaxed' | 'moderate' | 'packed'
export type BudgetLevel = 'budget' | 'moderate' | 'luxury'

export interface TripSetupRequest {
  destination: string
  start_date: string // YYYY-MM-DD
  end_date: string // YYYY-MM-DD
  interests: string[]
  daily_pace: DailyPace
  budget_level: BudgetLevel
  day_start_time: string // HH:MM
  day_end_time: string // HH:MM
  lang: 'en' | 'zh'
}

export interface ItineraryStop {
  attraction_name: string
  place_id: string
  // "attraction" or "meal" — a meal is a break with time reserved, no place picked.
  kind: 'attraction' | 'meal'
  arrival_offset_min: number
  depart_offset_min: number
  dwell_minutes: number
  // Travel time from the previous stop in visit order; 0 for the first stop.
  travel_from_prev_min: number
  // Minutes waiting at the door for this place to open (arrived early); 0 otherwise.
  wait_for_opening_min: number
}

export interface ItineraryDay {
  date: string
  stops: ItineraryStop[]
  weather_note: string | null
  // Attractions that were picked for this day but didn't make it in --
  // either they couldn't be fitted around opening hours and travel time, or
  // the place turns out to be closed all day that weekday. Already
  // localized copy lives in `notes`.
  dropped_stops: string[]
}

export interface TripPlanResponse {
  itinerary: ItineraryDay[]
  summary: string
  // Advisory lines (partial forecast, low-coverage destination, unresolved
  // constraints). Already localized by the backend.
  notes: string[]
  errors: string[]
}

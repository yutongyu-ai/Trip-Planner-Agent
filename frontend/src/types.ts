export type DailyPace = 'relaxed' | 'moderate' | 'packed'
export type BudgetLevel = 'budget' | 'moderate' | 'luxury'

export interface TripSetupRequest {
  destination: string
  start_date: string // YYYY-MM-DD
  end_date: string // YYYY-MM-DD
  interests: string[]
  preference_notes?: string
  daily_pace: DailyPace
  budget_level: BudgetLevel
  day_start_time: string // HH:MM
  day_end_time: string // HH:MM
  lang: 'en' | 'zh'
  avoid_outdoor_in_bad_weather: boolean
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
}

export interface TripPlanResponse {
  session_id: string
  itinerary: ItineraryDay[]
  summary: string
  // Advisory lines (partial forecast, low-coverage destination, unresolved
  // constraints). Already localized by the backend.
  notes: string[]
  errors: string[]
  // Stable tag for an error the frontend should act on (e.g. route back to
  // the setup form and point at a specific field) instead of just displaying
  // -- see backend/app/main.py. null for every other error, which is only
  // ever advisory alongside a real itinerary.
  error_code: string | null
}

export interface ChatResponse {
  reply: string
  itinerary: ItineraryDay[]
}

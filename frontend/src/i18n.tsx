import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'

export type Lang = 'en' | 'zh'

const translations = {
  en: {
    kicker: 'AI Trip Planner',
    title: 'Trip Planner',
    subtitle: "Tell us where you're headed and we'll build a day-by-day plan.",

    fieldDestination: 'Destination',
    destinationPlaceholder: 'e.g. London',
    sectionWhen: 'When',
    fieldStartDate: 'Start date',
    fieldEndDate: 'End date',
    fieldDayStart: 'Day starts at',
    fieldDayEnd: 'Day ends at',
    fieldInterests: 'Interests',
    interestsPlaceholder: 'e.g. museums, food, hiking',
    interestsHint: 'Comma-separated. Leave blank for general sightseeing.',
    sectionPreferences: 'Preferences',
    fieldPace: 'Daily pace',
    paceRelaxed: 'Relaxed',
    paceModerate: 'Moderate',
    pacePacked: 'Packed',
    fieldBudget: 'Budget',
    budgetBudget: 'Budget',
    budgetModerate: 'Moderate',
    budgetLuxury: 'Luxury',

    errorDestinationRequired: 'Enter a destination.',
    errorEndBeforeStart: 'End date must be on or after the start date.',
    errorDayEndBeforeStart: 'Day end time must be after the day start time.',
    errorDayStartInPast: "That start time has already passed today. Pick a later time or a different date.",

    submitting: 'Planning your trip…',
    submit: 'Plan my trip',

    loading: 'Planning your trip — please be patient while the agent checks attractions, weather, and routes…',
    genericError: 'Something went wrong. Please try again.',
    backToSetup: 'Back to trip setup',

    noStops: 'No stops planned for this day.',
    planAnother: 'Plan another trip',
    dayPrefix: 'Day ',
    daySuffix: '',
    visitDuration: 'visit',
    transit: 'transit',
    wait: 'wait',
    opensAt: 'opens',
    droppedPrefix: "Didn't fit the day: ",
    mealNear: 'time to eat — somewhere near {place}',

    errorNetwork: 'Could not reach the trip planner backend. Is it running?',
    errorTimeout: 'Trip planning is taking too long and was cancelled. Please try again.',
    errorRequestFailedPrefix: 'Trip planning failed',

    resultsNote:
      "Place names, weather, and summary text follow this toggle where the underlying data supports it. A place with no Chinese name on file (e.g. a small shop outside China) will still show its original name.",
  },
  zh: {
    kicker: 'AI 行程规划助手',
    title: '行程规划器',
    subtitle: '告诉我们你要去哪里，我们会为你生成逐日行程。',

    fieldDestination: '目的地',
    destinationPlaceholder: '例如：伦敦',
    sectionWhen: '时间',
    fieldStartDate: '开始日期',
    fieldEndDate: '结束日期',
    fieldDayStart: '每天开始时间',
    fieldDayEnd: '每天结束时间',
    fieldInterests: '兴趣',
    interestsPlaceholder: '例如：博物馆、美食、徒步',
    interestsHint: '用逗号分隔，留空表示常规观光。',
    sectionPreferences: '偏好',
    fieldPace: '节奏',
    paceRelaxed: '轻松',
    paceModerate: '适中',
    pacePacked: '紧凑',
    fieldBudget: '预算',
    budgetBudget: '经济',
    budgetModerate: '适中',
    budgetLuxury: '高端',

    errorDestinationRequired: '请输入目的地。',
    errorEndBeforeStart: '结束日期不能早于开始日期。',
    errorDayEndBeforeStart: '结束时间必须晚于开始时间。',
    errorDayStartInPast: '今天的开始时间已经过了，请选择更晚的时间或换一天。',

    submitting: '正在规划行程…',
    submit: '生成行程',

    loading: '正在规划你的行程——agent 需要查询景点、天气和路线，请耐心等待……',
    genericError: '出了点问题，请重试。',
    backToSetup: '返回重新设置',

    noStops: '这一天还没有安排行程。',
    planAnother: '规划另一趟旅行',
    dayPrefix: '第 ',
    daySuffix: ' 天',
    visitDuration: '游玩',
    transit: '通勤',
    wait: '等待',
    opensAt: '开门',
    droppedPrefix: '当天排不下：',
    mealNear: '吃饭时间——在 {place} 附近找一家',

    errorNetwork: '无法连接到行程规划后端，请确认后端服务是否已启动。',
    errorTimeout: '行程规划耗时过长，已取消，请重试。',
    errorRequestFailedPrefix: '行程规划失败',

    resultsNote: '摘要、天气和地点名称会尽量跟着这个开关走；如果某个地点在 Google 地图数据库里没有中文名称（比如国外的小商户），会保留原名。',
  },
} as const

export type TranslationKey = keyof (typeof translations)['en']

// Plain (non-hook) lookup for code outside the React tree -- api.ts isn't a
// component, so it can't call useLanguage(), but its thrown error messages
// still need to follow the same language toggle as everything else.
export function translate(lang: Lang, key: TranslationKey): string {
  return translations[lang][key]
}

interface LanguageContextValue {
  lang: Lang
  setLang: (lang: Lang) => void
  t: (key: TranslationKey) => string
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

export function LanguageProvider({ children }: { children: ReactNode }) {
  // Always starts in English, regardless of what a previous visit last left
  // the toggle on -- the app's primary audience (see README) reads English
  // first; a returning visitor can still switch with the toggle each time.
  const [lang, setLang] = useState<Lang>('en')

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const t = (key: TranslationKey) => translations[lang][key]

  return <LanguageContext.Provider value={{ lang, setLang, t }}>{children}</LanguageContext.Provider>
}

export function useLanguage(): LanguageContextValue {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider')
  }
  return context
}

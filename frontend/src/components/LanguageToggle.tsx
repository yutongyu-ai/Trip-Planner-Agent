import { useLanguage } from '../i18n'
import type { Lang } from '../i18n'

const OPTIONS: { value: Lang; label: string }[] = [
  { value: 'en', label: 'EN' },
  { value: 'zh', label: '中文' },
]

export function LanguageToggle() {
  const { lang, setLang } = useLanguage()

  return (
    <div className="lang-toggle" role="group" aria-label="Language">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={option.value === lang ? 'active' : ''}
          onClick={() => setLang(option.value)}
          aria-pressed={option.value === lang}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

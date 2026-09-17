import { useState } from 'react'
import type { FormEvent } from 'react'
import { TripPlanError, fetchChatReply } from '../api'
import { useLanguage } from '../i18n'
import type { ItineraryDay } from '../types'

interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
}

interface ChatEditorProps {
  sessionId: string
  onItineraryUpdate: (itinerary: ItineraryDay[]) => void
}

export function ChatEditor({ sessionId, onItineraryUpdate }: ChatEditorProps) {
  const { lang, t } = useLanguage()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const message = input.trim()
    if (!message || sending) return

    setSending(true)
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', text: message }])
    setInput('')

    try {
      const result = await fetchChatReply(sessionId, message, lang)
      setMessages((prev) => [...prev, { role: 'assistant', text: result.reply }])
      onItineraryUpdate(result.itinerary)
    } catch (err) {
      const text = err instanceof TripPlanError ? err.message : t('genericError')
      setError(`${t('chatErrorPrefix')}: ${text}`)
    } finally {
      setSending(false)
    }
  }

  return (
    <section className="chat-editor">
      <h3>{t('chatHeading')}</h3>
      <p className="chat-hint">{t('chatHint')}</p>

      {messages.length > 0 && (
        <ul className="chat-messages">
          {messages.map((message, i) => (
            <li key={i} className={`chat-message chat-message-${message.role}`}>
              {message.text}
            </li>
          ))}
        </ul>
      )}

      {error && <p className="form-error">{error}</p>}

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={t('chatPlaceholder')}
          disabled={sending}
        />
        <button type="submit" className="primary-button" disabled={sending || !input.trim()}>
          {sending ? t('chatSending') : t('chatSend')}
        </button>
      </form>
    </section>
  )
}

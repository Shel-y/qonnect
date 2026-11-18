import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
}

type Conversation = {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

type SavedProblem = {
  id: number
  content: string
  conversationId: string
  savedAt: number
}

type View = 'lab' | 'conversations' | 'saved'

type ChatResponse = {
  result?: string
  error?: string
}

const suggestions = [
  { icon: '⌘', title: 'Formular un problema', description: 'Define elementos, beneficios y presupuesto.', prompt: 'Ayúdame a formular un problema de selección.' },
  { icon: '◈', title: 'Optimización clásica', description: 'Explora métodos exactos y greedy.', prompt: 'Quiero resolver un problema mediante optimización clásica.' },
  { icon: '✳', title: 'Experimentar con QAOA', description: 'Explora la optimización cuántica.', prompt: 'Explícame cómo puedo experimentar con QAOA.' },
  { icon: '⟡', title: 'Comparar métodos', description: 'Combina estrategias clásicas y cuánticas.', prompt: 'Quiero comparar métodos clásicos y cuánticos.' },
]

function readStorage<T>(key: string): T {
  try {
    const value = localStorage.getItem(key)
    return value ? JSON.parse(value) as T : [] as T
  } catch {
    return [] as T
  }
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>(
    () => readStorage<Conversation[]>('qonnect-conversations'),
  )
  const [savedProblems, setSavedProblems] = useState<SavedProblem[]>(
    () => readStorage<SavedProblem[]>('qonnect-saved-problems'),
  )
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [view, setView] = useState<View>('lab')
  const [prompt, setPrompt] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')

  const [actorId] = useState(() => {
    const stored = localStorage.getItem('qonnect-actor-id')
    if (stored) return stored

    const id = crypto.randomUUID()
    localStorage.setItem('qonnect-actor-id', id)
    return id
  })

  const threadEndRef = useRef<HTMLDivElement>(null)

  const currentConversation = conversations.find((item) => item.id === currentId)
  const messages = currentConversation?.messages ?? []
  const hasThread = messages.length > 0

  useEffect(() => {
    localStorage.setItem('qonnect-conversations', JSON.stringify(conversations))
  }, [conversations])

  useEffect(() => {
    localStorage.setItem('qonnect-saved-problems', JSON.stringify(savedProblems))
  }, [savedProblems])

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const content = prompt.trim()
    if (!content || isSending) return

    setError('')
    setIsSending(true)

    const conversationId =
      currentId && currentConversation ? currentId : crypto.randomUUID()

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content,
    }

    const now = Date.now()

    if (currentId && currentConversation) {
      setConversations((items) =>
        items.map((item) =>
          item.id === conversationId
            ? {
                ...item,
                messages: [...item.messages, userMessage],
                updatedAt: now,
              }
            : item,
        ),
      )
    } else {
      setConversations((items) => [
        {
          id: conversationId,
          title: content.slice(0, 55),
          messages: [userMessage],
          updatedAt: now,
        },
        ...items,
      ])
      setCurrentId(conversationId)
    }

    setPrompt('')
    setView('lab')

    try {
      const apiUrl = import.meta.env.VITE_API_URL

      if (!apiUrl) {
        throw new Error(
          'No se encontró VITE_API_URL. Revisa la configuración de la API.',
        )
      }

      const response = await fetch(`${apiUrl.replace(/\/$/, '')}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: content,
          actor_id: actorId,
          session_id: conversationId,
        }),
      })

      let data: ChatResponse = {}
      const responseText = await response.text()

      try {
        const parsed: unknown = JSON.parse(responseText)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          data = parsed as ChatResponse
        }
      } catch {
        if (!response.ok) {
          throw new Error(`Error del servidor (${response.status}).`)
        }
        throw new Error('La API devolvió una respuesta no válida.')
      }

      if (!response.ok) {
        throw new Error(
          data.error || `Error del servidor (${response.status}).`,
        )
      }

      if (typeof data.result !== 'string' || !data.result.trim()) {
        throw new Error('El agente respondió, pero no devolvió un resultado válido.')
      }

      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.result,
      }

      setConversations((items) =>
        items.map((item) =>
          item.id === conversationId
            ? {
                ...item,
                messages: [...item.messages, assistantMessage],
                updatedAt: Date.now(),
              }
            : item,
        ),
      )
    } catch (err) {
      console.error('Error al comunicarse con Qonnect:', err)

      setConversations((items) =>
        items.flatMap((item) => {
          if (item.id !== conversationId) return [item]

          const messages = item.messages.filter(
            (message) => message.id !== userMessage.id,
          )

          return messages.length ? [{ ...item, messages }] : []
        }),
      )
      setPrompt(content)

      setError(
        err instanceof Error
          ? err.message
          : 'No fue posible comunicarse con Qonnect. Intenta nuevamente.',
      )
    } finally {
      setIsSending(false)
    }
  }

  const startNewConversation = () => {
    if (isSending) return
    setCurrentId(null)
    setPrompt('')
    setError('')
    setView('lab')
  }

  const openConversation = (id: string) => {
    if (isSending) return
    setCurrentId(id)
    setPrompt('')
    setError('')
    setView('lab')
  }

  const saveProblem = (message: Message) => {
    if (!currentId) return

    setSavedProblems((items) => {
      if (items.some((item) => item.id === message.id)) return items

      return [
        {
          id: message.id,
          content: message.content,
          conversationId: currentId,
          savedAt: Date.now(),
        },
        ...items,
      ]
    })
  }

  const removeSavedProblem = (id: number) => {
    setSavedProblems((items) => items.filter((item) => item.id !== id))
  }

  const chooseSuggestion = (value: string) => {
    setPrompt(value)
    setView('lab')
  }

  const formatDate = (timestamp: number) =>
    new Date(timestamp).toLocaleDateString('es-MX', {
      day: 'numeric',
      month: 'short',
    })

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">Q</div>
          <div>
            <div className="brand-name">Qonnect</div>
            <div className="brand-caption">QUANTUM OPTIMIZATION LAB</div>
          </div>
        </div>

        <button className="new-chat" onClick={startNewConversation} disabled={isSending}>
          <span>＋</span> Nueva conversación
        </button>

        <div className="nav-section">
          <div className="nav-label">WORKSPACE</div>
          <button
            className={`nav-item ${view === 'lab' ? 'active' : ''}`}
            onClick={() => setView('lab')}
          >
            <span className="nav-icon">⌘</span> Laboratorio
          </button>
          <button
            className={`nav-item ${view === 'conversations' ? 'active' : ''}`}
            onClick={() => setView('conversations')}
          >
            <span className="nav-icon">◷</span> Conversaciones
          </button>
          <button
            className={`nav-item ${view === 'saved' ? 'active' : ''}`}
            onClick={() => setView('saved')}
          >
            <span className="nav-icon">▦</span> Problemas guardados
          </button>
        </div>

        <div className="sidebar-bottom">
          <div className="experiment-card">
            <div className="experiment-icon">✳</div>
            <div>
              <strong>Hybrid thinking</strong>
              <p>Classical meets quantum.</p>
            </div>
          </div>
          <div className="sidebar-footer">
            <div className="avatar">Q</div>
            <div className="footer-user">
              <strong>Guest workspace</strong>
              <span>Sesión de demostración</span>
            </div>
            <span className="more">···</span>
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="crumb-separator">/</span>
            <strong>
              {view === 'lab'
                ? 'Laboratorio'
                : view === 'conversations'
                  ? 'Conversaciones'
                  : 'Problemas guardados'}
            </strong>
          </div>
          <div className="topbar-right">
            <div className="connection-status">
              <span className="status-dot" />
              AgentCore · API configurada
            </div>
            <button className="help-button" aria-label="Ayuda">?</button>
          </div>
        </header>

        <div className="workspace">
          {view === 'conversations' ? (
            <section className="library-view">
              <div className="library-heading">
                <div className="eyebrow">
                  <span className="eyebrow-line" /> YOUR WORKSPACE
                </div>
                <h1>Conversaciones</h1>
                <p>Retoma una sesión anterior o empieza una nueva.</p>
              </div>

              {conversations.length === 0 ? (
                <div className="empty-state">
                  <span>◷</span>
                  <h2>Aún no hay conversaciones</h2>
                  <p>Cuando envíes tu primer mensaje, aparecerá aquí.</p>
                  <button className="primary-action" onClick={startNewConversation}>
                    Iniciar conversación
                  </button>
                </div>
              ) : (
                <div className="library-list">
                  {[...conversations]
                    .sort((a, b) => b.updatedAt - a.updatedAt)
                    .map((item) => (
                      <button
                        className="library-item"
                        key={item.id}
                        onClick={() => openConversation(item.id)}
                        disabled={isSending}
                      >
                        <span className="library-item-icon">◷</span>
                        <span className="library-item-content">
                          <strong>{item.title}</strong>
                          <small>{item.messages.length} mensaje(s)</small>
                        </span>
                        <time>{formatDate(item.updatedAt)}</time>
                        <span className="library-arrow">↗</span>
                      </button>
                    ))}
                </div>
              )}
            </section>
          ) : view === 'saved' ? (
            <section className="library-view">
              <div className="library-heading">
                <div className="eyebrow">
                  <span className="eyebrow-line" /> YOUR WORKSPACE
                </div>
                <h1>Problemas guardados</h1>
                <p>Mensajes que has marcado para consultar después.</p>
              </div>

              {savedProblems.length === 0 ? (
                <div className="empty-state">
                  <span>▦</span>
                  <h2>Tu colección está vacía</h2>
                  <p>
                    En una conversación, utiliza «Guardar problema» para conservar
                    un mensaje.
                  </p>
                  <button className="primary-action" onClick={startNewConversation}>
                    Ir al laboratorio
                  </button>
                </div>
              ) : (
                <div className="library-list">
                  {savedProblems.map((item) => (
                    <article className="saved-item" key={item.id}>
                      <p>{item.content}</p>
                      <div className="saved-item-actions">
                        <button onClick={() => openConversation(item.conversationId)}>
                          Abrir conversación ↗
                        </button>
                        <button onClick={() => removeSavedProblem(item.id)}>
                          Eliminar
                        </button>
                        <time>{formatDate(item.savedAt)}</time>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>
          ) : (
            <div className={`welcome${hasThread ? ' has-thread' : ''}`}>
              {!hasThread ? (
                <>
                  <div className="eyebrow">
                    <span className="eyebrow-line" /> YOUR OPTIMIZATION WORKSPACE
                  </div>
                  <div className="orbital-mark">
                    <div className="orbit orbit-one" />
                    <div className="orbit orbit-two" />
                    <div className="orbit orbit-three" />
                    <div className="orbit-core">Q</div>
                    <span className="orbit-dot dot-one" />
                    <span className="orbit-dot dot-two" />
                  </div>
                  <h1>Ideas into <span>optimization.</span></h1>
                  <p className="welcome-copy">
                    Explora problemas de selección y descubre cómo los métodos
                    clásicos, cuánticos e híbridos pueden ayudarte a resolverlos.
                  </p>
                  <div className="suggestion-grid">
                    {suggestions.map((item) => (
                      <button
                        className="suggestion-card"
                        key={item.title}
                        onClick={() => chooseSuggestion(item.prompt)}
                      >
                        <span className="suggestion-icon">{item.icon}</span>
                        <span className="suggestion-title">{item.title}</span>
                        <span className="suggestion-description">
                          {item.description}
                        </span>
                        <span className="suggestion-arrow">↗</span>
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <div className="thread" aria-live="polite">
                  {messages.map((message) => (
                    <article className={`message ${message.role}`} key={message.id}>
                      <div className="message-author">
                        {message.role === 'user' ? 'TÚ' : 'QONNECT'}
                      </div>
                      {message.role === 'assistant' ? (
                        <div className="message-content markdown-content">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p>{message.content}</p>
                      )}
                      {message.role === 'user' && (
                        <button
                          className="save-problem"
                          onClick={() => saveProblem(message)}
                        >
                          {savedProblems.some((item) => item.id === message.id)
                            ? '✓ Guardado'
                            : '＋ Guardar problema'}
                        </button>
                      )}
                    </article>
                  ))}
                  {isSending && (
                    <article className="message assistant">
                      <div className="message-author">QONNECT</div>
                      <p>Qonnect está pensando...</p>
                    </article>
                  )}
                  <div ref={threadEndRef} />
                </div>
              )}

              <div className="composer">
                <textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void sendMessage()
                    }
                  }}
                  placeholder="Cuéntale a Qonnect qué problema quieres resolver..."
                  rows={2}
                  aria-label="Mensaje para Qonnect"
                  disabled={isSending}
                />
                <div className="composer-bottom">
                  <span className="composer-hint">
                    <span>↵</span> Enter para enviar · Shift + Enter para nueva línea
                  </span>
                  <button
                    className="send-button"
                    disabled={!prompt.trim() || isSending}
                    onClick={() => void sendMessage()}
                    aria-label="Enviar mensaje"
                  >
                    {isSending ? '…' : '↑'}
                  </button>
                </div>
              </div>

              {error && (
                <div className="preview-notice" role="alert">
                  ⚠ {error}
                </div>
              )}

              {hasThread && !error && (
                <div className="preview-notice">
                  ⓘ Tus conversaciones se guardan en este navegador.
                </div>
              )}

              {!hasThread && (
                <div className="capabilities">
                  <span><i className="cap-dot classical" /> Classical methods</span>
                  <span><i className="cap-dot quantum" /> Quantum experiments</span>
                  <span><i className="cap-dot hybrid" /> Hybrid optimization</span>
                </div>
              )}
            </div>
          )}

          <footer className="workspace-footer">
            QONNECT <span>·</span> EXPERIMENT, CONNECT, OPTIMIZE
          </footer>
        </div>
      </section>
    </main>
  )
}

export default App

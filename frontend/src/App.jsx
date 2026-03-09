import { useState, useCallback, useRef } from 'react'

const API = 'http://localhost:8000'
const STEPS = ['Upload', 'Transcribe', 'Analyze', 'Complete']

const TYPE_STYLES = {
  commitment: { bg: '#7c3aed1a', text: '#c4b5fd', border: '#7c3aed40', dot: '#a78bfa', label: 'Commitment' },
  request:    { bg: '#d9770620', text: '#fcd34d', border: '#d9770640', dot: '#fbbf24', label: 'Request' },
  information:{ bg: '#0284c720', text: '#7dd3fc', border: '#0284c740', dot: '#38bdf8', label: 'Info' },
}

function fmtDate(iso) {
  if (!iso) return null
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}
function fmtSize(b) {
  return b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`
}

// ── Icons ──────────────────────────────────────────────────────────────────────
const UploadIcon = ({ size = 40, color = '#64748b' }) => (
  <svg width={size} height={size} fill="none" stroke={color} strokeWidth="1.4" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
)
const CheckIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
  </svg>
)
const SpinIcon = ({ size = 16 }) => (
  <svg width={size} height={size} fill="none" viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.2 }} />
    <path fill="currentColor" d="M4 12a8 8 0 018-8v8z" style={{ opacity: 0.8 }} />
  </svg>
)

// ── Step Progress ──────────────────────────────────────────────────────────────
function StepProgress({ currentStep }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '2.5rem' }}>
      {STEPS.map((step, i) => (
        <div key={step} style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{
              width: 36, height: 36, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 13, fontWeight: 600,
              transition: 'all 0.4s',
              background: i < currentStep ? '#7c3aed' : i === currentStep ? '#6d28d9' : '#1e293b',
              color: i <= currentStep ? '#fff' : '#475569',
              border: i === currentStep ? '3px solid #7c3aed50' : i < currentStep ? 'none' : '1.5px solid #334155',
              boxShadow: i === currentStep ? '0 0 0 4px #7c3aed20' : i < currentStep ? '0 4px 12px #7c3aed40' : 'none',
            }}>
              {i < currentStep ? <CheckIcon /> : i + 1}
            </div>
            <span style={{
              fontSize: 11, marginTop: 8, fontWeight: 500, letterSpacing: '0.03em',
              color: i <= currentStep ? '#cbd5e1' : '#475569',
              transition: 'color 0.3s',
            }}>{step}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div style={{
              width: 72, height: 1, marginBottom: 20, marginLeft: 8, marginRight: 8,
              background: i < currentStep ? '#7c3aed' : '#1e293b',
              transition: 'background 0.6s',
            }} />
          )}
        </div>
      ))}
    </div>
  )
}

// ── Upload Zone ────────────────────────────────────────────────────────────────
function UploadZone({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  const over  = (e) => { e.preventDefault(); if (!disabled) setDragging(true) }
  const leave = () => setDragging(false)

  const base = {
    border: `2px dashed ${dragging ? '#818cf8' : '#1e293b'}`,
    borderRadius: 20,
    padding: '56px 40px',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: 12,
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    background: dragging ? '#6366f115' : 'transparent',
    transition: 'all 0.25s',
    transform: dragging ? 'scale(1.01)' : 'scale(1)',
  }

  return (
    <div onClick={() => !disabled && inputRef.current?.click()}
      onDrop={handleDrop} onDragOver={over} onDragLeave={leave} style={base}>
      <input ref={inputRef} type="file" style={{ display: 'none' }}
        accept=".wav,.mp3,.m4a,.mp4,.webm,.ogg"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        disabled={disabled} />

      <div style={{
        width: 80, height: 80, borderRadius: 20,
        background: dragging ? '#6366f125' : '#0f172a',
        border: `1.5px solid ${dragging ? '#818cf840' : '#1e293b'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all 0.25s',
        transform: dragging ? 'scale(1.08)' : 'scale(1)',
      }}>
        <UploadIcon size={36} color={dragging ? '#818cf8' : '#475569'} />
      </div>

      <div style={{ textAlign: 'center' }}>
        <p style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 17, marginBottom: 6 }}>
          {dragging ? 'Release to upload' : 'Drop your meeting recording here'}
        </p>
        <p style={{ color: '#475569', fontSize: 13 }}>
          Supports&nbsp;
          <span style={{ color: '#94a3b8' }}>.wav · .mp3 · .m4a · .mp4</span>
        </p>
      </div>

      <div style={{
        marginTop: 4,
        padding: '8px 20px',
        borderRadius: 999,
        border: '1px solid #1e293b',
        color: '#64748b',
        fontSize: 12,
        background: '#0f172a',
      }}>
        or click to browse
      </div>
    </div>
  )
}

// ── Processing Card ────────────────────────────────────────────────────────────
function ProcessingCard({ file, status }) {
  return (
    <div style={{
      background: '#0f172a', border: '1px solid #1e293b', borderRadius: 16,
      padding: '20px 24px', marginTop: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 12,
          background: '#6d28d920', border: '1px solid #7c3aed30',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          <UploadIcon size={18} color="#a78bfa" />
        </div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <p style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 500,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file?.name}</p>
          <p style={{ color: '#475569', fontSize: 11, marginTop: 2 }}>{file ? fmtSize(file.size) : ''}</p>
        </div>
      </div>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        background: '#1e293b', borderRadius: 12, padding: '12px 16px',
      }}>
        <SpinIcon size={15} />
        <span style={{ color: '#94a3b8', fontSize: 13 }}>{status}</span>
      </div>
    </div>
  )
}

// ── Task Card ──────────────────────────────────────────────────────────────────
function TaskCard({ task }) {
  const s = TYPE_STYLES[task.type] || TYPE_STYLES.information
  return (
    <div style={{
      background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14,
      padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12,
      transition: 'border-color 0.2s',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <p style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.6, flex: 1 }}>{task.sentence}</p>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '3px 10px', borderRadius: 999, fontSize: 11, fontWeight: 600,
          background: s.bg, color: s.text, border: `1px solid ${s.border}`, flexShrink: 0,
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: s.dot }} />
          {s.label}
        </span>
      </div>

      {(task.owner || task.task || task.deadline) && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '8px 20px',
          borderTop: '1px solid #1e293b', paddingTop: 12, fontSize: 12,
        }}>
          {task.owner && (
            <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 5 }}>
              <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              {task.owner}
            </span>
          )}
          {task.task && (
            <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 5, textTransform: 'capitalize' }}>
              <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {task.task}
            </span>
          )}
          {task.deadline && (
            <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: 5 }}>
              <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {fmtDate(task.deadline)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// ── Meeting History ────────────────────────────────────────────────────────────
function MeetingHistory({ meetings, onSelect }) {
  if (!meetings.length) return (
    <p style={{ textAlign: 'center', color: '#334155', fontSize: 13, padding: '24px 0' }}>
      No past meetings found.
    </p>
  )
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {meetings.map((m) => (
        <button key={m.id} onClick={() => onSelect(m)} style={{
          background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14,
          padding: '14px 18px', textAlign: 'left', cursor: 'pointer', width: '100%',
          transition: 'border-color 0.2s, background 0.2s',
        }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = '#7c3aed50'; e.currentTarget.style.background = '#1e293b' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = '#1e293b'; e.currentTarget.style.background = '#0f172a' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ color: '#a78bfa', fontSize: 11, fontWeight: 600 }}>
              {m.name || (m.created_at ? new Date(m.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : `Meeting #${m.id}`)}
            </span>
            <span style={{ color: '#334155', fontSize: 11 }}>
              {m.created_at ? new Date(m.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''}
            </span>
          </div>
          <p style={{
            color: '#94a3b8', fontSize: 13, lineHeight: 1.5, marginBottom: 6,
            display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
          }}>
            {m.summary || 'No summary available'}
          </p>
          <p style={{ color: '#334155', fontSize: 11 }}>
            {(m.tasks_json || []).length} action item{(m.tasks_json || []).length !== 1 ? 's' : ''}
          </p>
        </button>
      ))}
    </div>
  )
}

// ── Section Header ─────────────────────────────────────────────────────────────
function SectionHeader({ icon, title, badge, action }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
      <div style={{
        width: 30, height: 30, borderRadius: 8,
        background: '#7c3aed20', border: '1px solid #7c3aed30',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        {icon}
      </div>
      <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: 15 }}>{title}</span>
      {badge != null && (
        <span style={{
          background: '#7c3aed20', color: '#c4b5fd', border: '1px solid #7c3aed30',
          fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 999,
        }}>{badge}</span>
      )}
      {action && <div style={{ marginLeft: 'auto' }}>{action}</div>}
    </div>
  )
}

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [step, setStep] = useState(0)
  const [file, setFile] = useState(null)
  const [meetingName, setMeetingName] = useState('')
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [meetings, setMeetings] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)

  const loadHistory = async () => {
    setHistoryLoading(true)
    try {
      const res = await fetch(`${API}/meetings`)
      if (res.ok) setMeetings(await res.json())
    } catch { /* backend offline */ }
    finally { setHistoryLoading(false) }
  }

  const handleFile = async (f) => {
    setFile(f); setError(''); setResult(null); setShowHistory(false)
    setStep(1); setStatus('Transcribing audio with Whisper...')
    const form = new FormData()
    form.append('file', f)
    form.append('name', meetingName.trim())
    try {
      const r1 = await fetch(`${API}/transcribe`, { method: 'POST', body: form })
      if (!r1.ok) throw new Error((await r1.json().catch(() => ({}))).detail || 'Transcription failed')
      const { id, transcript, name } = await r1.json()

      setStep(2); setStatus('Running NLP pipeline — extracting tasks & summary...')

      const r2 = await fetch(`${API}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, text: transcript }),
      })
      if (!r2.ok) throw new Error((await r2.json().catch(() => ({}))).detail || 'NLP extraction failed')
      const data = await r2.json()

      setStep(3); setStatus(''); setResult({ ...data, transcript, name }); loadHistory()
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?')
      setStep(0); setStatus('')
    }
  }

  const reset = () => { setStep(0); setFile(null); setResult(null); setError(''); setStatus(''); setShowHistory(false); setMeetingName('') }

  const handleToggleHistory = async () => {
    if (!showHistory && meetings.length === 0) await loadHistory()
    setShowHistory(v => !v)
  }

  const handleSelectMeeting = (m) => {
    setResult({ id: m.id, name: m.name || '', summary: m.summary, tasks: m.tasks_json || [] })
    setStep(3); setError(''); setShowHistory(false)
  }

  return (
    <div style={{ minHeight: '100vh', background: 'radial-gradient(ellipse 80% 60% at 50% 0%, #0f0a2e 0%, #020617 60%)' }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      <div style={{ maxWidth: 680, margin: '0 auto', padding: '56px 24px 80px' }}>

        {/* ── Header ── */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            background: '#7c3aed15', border: '1px solid #7c3aed30',
            borderRadius: 999, padding: '6px 16px', marginBottom: 20,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: '#a78bfa',
              animation: 'pulse 2s infinite',
            }} />
            <span style={{ color: '#c4b5fd', fontSize: 12, fontWeight: 500 }}>
              AI-Powered · Whisper + spaCy + BART
            </span>
          </div>
          <h1 style={{
            fontSize: 40, fontWeight: 800, color: '#f8fafc',
            letterSpacing: '-0.03em', lineHeight: 1.15, marginBottom: 12,
          }}>
            Meeting Action Tracker
          </h1>
          <p style={{ color: '#64748b', fontSize: 15, lineHeight: 1.6 }}>
            Upload a recording. Get a summary and action items instantly.
          </p>
        </div>

        {/* ── Steps ── */}
        <StepProgress currentStep={step} />

        {/* ── Upload / Processing ── */}
        {step < 3 && (
          <div>
            {step === 0 && (
              <input
                type="text"
                placeholder="Meeting name (optional)"
                value={meetingName}
                onChange={e => setMeetingName(e.target.value)}
                style={{
                  width: '100%', marginBottom: 12,
                  background: '#0f172a', border: '1px solid #1e293b',
                  borderRadius: 12, padding: '12px 16px',
                  color: '#e2e8f0', fontSize: 14, outline: 'none',
                  transition: 'border-color 0.2s',
                }}
                onFocus={e => e.target.style.borderColor = '#7c3aed60'}
                onBlur={e => e.target.style.borderColor = '#1e293b'}
              />
            )}
            <UploadZone onFile={handleFile} disabled={step > 0} />

            {step > 0 && <ProcessingCard file={file} status={status} />}

            {error && (
              <div style={{
                marginTop: 16, display: 'flex', alignItems: 'flex-start', gap: 12,
                background: '#ef444415', border: '1px solid #ef444430',
                borderRadius: 14, padding: '14px 18px',
              }}>
                <svg style={{ flexShrink: 0, marginTop: 1 }} width="16" height="16" fill="none" stroke="#f87171" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p style={{ color: '#fca5a5', fontSize: 13 }}>{error}</p>
              </div>
            )}

            {step === 0 && (
              <button onClick={handleToggleHistory} style={{
                marginTop: 20, width: '100%', background: 'none', border: 'none',
                color: '#334155', fontSize: 13, cursor: 'pointer', padding: '10px 0',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                transition: 'color 0.2s',
              }}
                onMouseEnter={e => e.currentTarget.style.color = '#94a3b8'}
                onMouseLeave={e => e.currentTarget.style.color = '#334155'}>
                {historyLoading
                  ? <><SpinIcon size={13} /> Loading history...</>
                  : <>{showHistory ? '↑ Hide' : '↓ View'} past meetings</>}
              </button>
            )}
          </div>
        )}

        {/* ── Results ── */}
        {result && step === 3 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Summary */}
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 18, padding: '24px 28px' }}>
              <SectionHeader
                icon={<svg width="14" height="14" fill="none" stroke="#a78bfa" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>}
                title={result.name || 'Summary'}
                action={
                  <button onClick={reset} style={{
                    background: '#1e293b', border: '1px solid #334155', borderRadius: 8,
                    color: '#64748b', fontSize: 12, padding: '6px 14px', cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                    onMouseEnter={e => { e.currentTarget.style.color = '#a78bfa'; e.currentTarget.style.borderColor = '#7c3aed60' }}
                    onMouseLeave={e => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = '#334155' }}>
                    + New Meeting
                  </button>
                }
              />
              <p style={{ color: '#94a3b8', fontSize: 14, lineHeight: 1.75 }}>
                {result.summary || 'No summary was generated for this meeting.'}
              </p>
            </div>

            {/* Action Items */}
            <div>
              <SectionHeader
                icon={<svg width="14" height="14" fill="none" stroke="#a78bfa" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>}
                title="Action Items"
                badge={(result.tasks || []).length}
              />
              {(result.tasks || []).length === 0 ? (
                <div style={{
                  background: '#0f172a', border: '1px solid #1e293b', borderRadius: 16,
                  padding: '48px 24px', textAlign: 'center', color: '#334155', fontSize: 14,
                }}>
                  No action items detected in this meeting.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {(result.tasks || []).map((task, i) => <TaskCard key={i} task={task} />)}
                </div>
              )}
            </div>

            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: 24, paddingTop: 4 }}>
              {Object.values(TYPE_STYLES).map(s => (
                <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#475569' }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: s.dot }} />
                  {s.label}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── History ── */}
        {showHistory && step === 0 && (
          <div style={{ marginTop: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <span style={{ color: '#334155', fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Past Meetings
              </span>
              <div style={{ flex: 1, height: 1, background: '#1e293b' }} />
            </div>
            <MeetingHistory meetings={meetings} onSelect={handleSelectMeeting} />
          </div>
        )}

      </div>
    </div>
  )
}

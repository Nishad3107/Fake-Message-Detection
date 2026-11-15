import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const prediction = {
  text: 'A public report describes the latest event and evidence.', cleaned_text: 'a public report describes the latest event and evidence.', prediction: 'unverified' as const, classification: 'unverified' as const, display_label: 'Model prediction: unverified', confidence: 0.556979, confidence_category: 'low_confidence' as const, class_probabilities: { false: 0.219515, true: 0.223506, unverified: 0.556979 }, detected_features: ['evidence', 'report'], detected_keywords: ['evidence'], explanation: 'The model found statistical language patterns.', explanation_details: { summary: 'The model found statistical language patterns.' }, influential_terms: ['evidence'], low_confidence: true, model_name: 'PHEME tuned Multinomial Naive Bayes', model_version: 'pheme-tuned-multinomial-naive-bayes-v1', training_dataset_reference: 'PHEME', training_split_reference: 'backend/data/splits/pheme_split_manifest.json', model_limitations: 'Not factual verification.', timestamp: new Date().toISOString(), warning: 'This is a probabilistic NLP prediction, not factual verification.', disclaimer: 'This is not factual verification.',
}

describe('academic frontend', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/')
    vi.restoreAllMocks()
  })

  it('renders the landing page and academic CTAs', () => {
    render(<App />)
    expect(screen.getByText(/Understand the signal/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /Analyze a Message/i }).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/does not verify factual truth/i).length).toBeGreaterThan(0)
  })

  it('validates analyzer input before calling the API', async () => {
    window.location.hash = '#analyze'
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /Analyze post/i }))
    expect(screen.getByText(/at least 10 characters/i)).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('renders privacy notice and treats script-like input as text', () => {
    window.location.hash = '#analyze'
    render(<App />)
    const input = '<script>alert("xss")</script>'
    fireEvent.change(screen.getByLabelText(/Social media post/i), { target: { value: input } })
    expect(screen.getByDisplayValue(input)).toBeInTheDocument()
    expect(screen.getByText(/stored in local SQLite history/i)).toBeInTheDocument()
    expect(document.querySelector('script:not([src])')).toBeNull()
  })

  it('renders a real prediction response and uncertainty warning', async () => {
    window.location.hash = '#analyze'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => prediction }))
    render(<App />)
    fireEvent.change(screen.getByLabelText(/Social media post/i), { target: { value: prediction.text } })
    fireEvent.click(screen.getByRole('button', { name: /Analyze post/i }))
    await waitFor(() => expect(screen.getByText(/Model prediction: unverified/i)).toBeInTheDocument())
    expect(screen.getAllByText(/probabilistic NLP prediction/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText('56%').length).toBeGreaterThan(0)
    expect(screen.getByText('evidence')).toBeInTheDocument()
  })

  it('renders methodology and research results from the single data source', () => {
    window.location.hash = '#methodology'
    render(<App />)
    expect(screen.getByText(/dataset and labels/i)).toBeInTheDocument()
    expect(screen.getByText('0.211379')).toBeInTheDocument()
    window.location.hash = '#results'
    fireEvent(window, new HashChangeEvent('hashchange'))
    expect(screen.getByText(/The honest scoreboard/i)).toBeInTheDocument()
    expect(screen.getByText(/true → unverified/i)).toBeInTheDocument()
  })

  it('renders an empty history state from the history API', async () => {
    window.location.hash = '#dashboard'
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ items: [], summary: { total_analyses: 0, true: 0, false: 0, unverified: 0, low_confidence: 0, average_confidence: 0 } }) }))
    render(<App />)
    await waitFor(() => expect(screen.getByText(/No analyses yet/i)).toBeInTheDocument())
    expect(screen.getAllByText(/Analyze a message/i).length).toBeGreaterThan(0)
  })
})

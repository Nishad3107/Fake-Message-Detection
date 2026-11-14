import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('application shell', () => {
  it('renders the landing page and analyzer call to action', () => {
    window.history.pushState({}, '', '/')
    render(<App />)
    expect(screen.getByText(/Read the signal/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /analyze a post/i }).length).toBeGreaterThan(0)
  })
})

export type Prediction = { id?: number; text?: string; classification: 'misinformation' | 'likely_true'; confidence: number; detected_keywords: string[]; explanation: string; influential_terms: string[]; low_confidence: boolean; model_name: string; model_version: string; timestamp: string; disclaimer: string }
export type HistoryResponse = { items: Prediction[]; summary: { total_analyses: number; likely_true: number; misinformation: number; average_confidence: number } }
export type ModelInfo = { name: string; version: string; algorithm: string; dataset: string; metrics: Record<string, unknown>; ready: boolean }
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
export async function predictPost(text: string): Promise<Prediction> {
  const response = await fetch(`${API_URL}/api/v1/predict`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text}) })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? 'Unable to analyze this post.') }
  return response.json() as Promise<Prediction>
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`)
  if (!response.ok) throw new Error('Unable to reach the analysis service.')
  return response.json() as Promise<T>
}

export const getHistory = () => get<HistoryResponse>('/api/v1/history')
export const getModelInfo = () => get<ModelInfo>('/api/v1/model-info')

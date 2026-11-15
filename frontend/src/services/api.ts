export type Label = 'true' | 'false' | 'unverified'
export type Prediction = { id?: number; text?: string; cleaned_text?: string; prediction: Label; classification: Label; display_label: string; confidence: number; confidence_category: 'low_confidence' | 'moderate_confidence' | 'high_confidence'; class_probabilities: Record<Label, number>; detected_features: string[]; detected_keywords: string[]; explanation: string; explanation_details: { summary?: string; important_terms?: string[]; limitations?: string }; influential_terms: string[]; low_confidence: boolean; model_name: string; model_version: string; training_dataset_reference?: string | string[]; training_split_reference?: string; model_limitations: string; warning: string; timestamp: string; disclaimer: string }
export type HistoryResponse = { items: Prediction[]; summary: { total_analyses: number; true: number; false: number; unverified: number; low_confidence: number; average_confidence: number } }
export type ModelInfo = { name: string; version: string; algorithm: string; dataset: string; dataset_reference?: string; labels: Label[]; number_of_classes: number; tfidf_configuration: Record<string, unknown>; model_hyperparameters: Record<string, unknown>; final_test_macro_f1?: number; limitations: string; metrics: Record<string, unknown>; ready: boolean }
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
export const getMetrics = () => get<Record<string, unknown>>('/api/v1/metrics')

/**
 * Sentiment Analysis API Service
 * Handles all API calls to the backend sentiment analysis service.
 */

const API_BASE_URL = 'http://localhost:8000'

export interface SocialFeatures {
  emoji_count: number
  emoji_descriptions: string[]
  hashtag_count: number
  hashtags: string[]
  mention_count: number
  mentions: string[]
  url_count: number
  exclamation_count: number
  question_count: number
  cleaned_text: string
}

export interface AnalyzeResponse {
  sentiment: 'positive' | 'negative' | 'neutral'
  confidence: number
  confidence_per_class: {
    positive: number
    negative: number
    neutral: number
  }
  social_features: SocialFeatures
  raw_text: string
}

export interface AnalyzeError {
  detail: string
}

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(errorData.detail || `HTTP error ${response.status}`)
  }

  return response.json()
}

export async function getLabels(): Promise<{ labels: string[] }> {
  const response = await fetch(`${API_BASE_URL}/api/labels`)
  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}`)
  }
  return response.json()
}

export async function checkHealth(): Promise<{ status: string; model_loaded: string; tokenizer_loaded: string }> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}`)
  }
  return response.json()
}

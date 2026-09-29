export type Probability = { label: string; value: number }

export type AnalysisResult = {
  id: string
  title: string
  subtitle: string
  foul: Probability[]
  severity: Probability[]
  temporalWeights: number[]
  viewWeights: number[]
  confidence: 'low' | 'moderate' | 'high'
  note: string
  videos?: string[]
}

export type VideoInput = {
  id: string
  file: File
  url: string
  duration?: number
  status: 'checking' | 'valid' | 'invalid'
  error?: string
}

export type AnalysisStatus = 'idle' | 'ready' | 'analyzing' | 'completed' | 'failed'

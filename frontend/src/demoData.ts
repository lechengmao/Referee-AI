import type { AnalysisResult } from './types'

export const demoCases: AnalysisResult[] = [
  {
    id: 'case-01', title: 'Incident 01', subtitle: '3 synchronized views · 5.0 s',
    videos: ['/demo/incident-01/clip_0.mp4', '/demo/incident-01/clip_1.mp4', '/demo/incident-01/clip_2.mp4'],
    foul: [{ label: 'Tackling', value: .7246 }, { label: 'Challenge', value: .0926 }, { label: 'Standing tackling', value: .0890 }],
    severity: [{ label: 'Offence + Yellow card', value: .6567 }, { label: 'Offence + No card', value: .1548 }, { label: 'Offence + Red card', value: .1200 }],
    temporalWeights: [.2031, .2386, .1556, .2519, .1509], viewWeights: [.2395, .4157, .3447], confidence: 'moderate',
    note: 'The model prediction matches the official labels for this prepared example: tackling and a yellow-card offence.',
  },
  {
    id: 'case-02', title: 'Incident 02', subtitle: '4 synchronized views · 5.0 s',
    videos: ['/demo/incident-02/clip_0.mp4', '/demo/incident-02/clip_1.mp4', '/demo/incident-02/clip_2.mp4', '/demo/incident-02/clip_3.mp4'],
    foul: [{ label: 'Standing tackling', value: .2826 }, { label: 'Tackling', value: .2492 }, { label: 'Holding', value: .1565 }],
    severity: [{ label: 'Offence + No card', value: .3384 }, { label: 'Offence + Yellow card', value: .3085 }, { label: 'No offence', value: .1978 }],
    temporalWeights: [.1997, .1926, .2020, .2002, .2054], viewWeights: [.2301, .2591, .2535, .2572], confidence: 'low',
    note: 'This is an ambiguous four-view example. The small probability margins show why the original footage must stay central.',
  },
  {
    id: 'case-03', title: 'Incident 03', subtitle: '2 synchronized views · 5.0 s',
    videos: ['/demo/incident-03/clip_0.mp4', '/demo/incident-03/clip_1.mp4'],
    foul: [{ label: 'Standing tackling', value: .5617 }, { label: 'Tackling', value: .2753 }, { label: 'Holding', value: .1012 }],
    severity: [{ label: 'Offence + Yellow card', value: .5916 }, { label: 'Offence + No card', value: .1942 }, { label: 'Offence + Red card', value: .1639 }],
    temporalWeights: [.1750, .1773, .1848, .2336, .2293], viewWeights: [.3977, .6023], confidence: 'moderate',
    note: 'The model prediction matches the official labels for this prepared example: standing tackling and a yellow-card offence.',
  },
]

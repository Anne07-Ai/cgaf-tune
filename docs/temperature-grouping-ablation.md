# Temperature × grouping generation ablation

Nine CGAF configurations completed 100 steps on Dolly-15k with seed 42.

| Rank | Grouping | Temperature | Domain F1 | Retention F1 | Harmonic F1 | Time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Layer** | **0.03** | 0.3272 | 0.2168 | **0.2608** | 113.5 s |
| 2 | Module | 0.10 | 0.3236 | 0.2171 | 0.2599 | 134.1 s |
| 3 | Module | 0.03 | 0.3198 | 0.2184 | 0.2595 | 133.6 s |
| 4 | Layer | 0.10 | 0.3272 | 0.2116 | 0.2570 | 113.6 s |
| 5 | Layer | 0.30 | 0.3272 | 0.2080 | 0.2544 | 114.9 s |
| 6 | Global | 0.30 | 0.3272 | 0.2076 | 0.2541 | 112.2 s |
| 7 | Global | 0.03 | 0.3170 | 0.2089 | 0.2518 | 115.8 s |
| 8 | Module | 0.30 | 0.3236 | 0.2034 | 0.2497 | 133.0 s |
| 9 | Global | 0.10 | 0.3272 | 0.2006 | 0.2487 | 112.3 s |

The frozen base harmonic F1 was 0.1699. Layer/0.03 improved it to 0.2608 and is the practical
screening winner. Module/0.10 was only 0.0009 lower but took about 18% longer and generated roughly
15 times more projection operations than layer grouping.

All exact-match scores were zero, which is expected to be overly strict for free-form
summarization but also means this ranking relies entirely on token F1. The evaluation contains
only 12 examples per objective and one seed. This is configuration screening, not confirmatory
evidence.

[Hugging Face job](https://huggingface.co/jobs/lakshmianne/6aa2ada521047bf1b037291a)

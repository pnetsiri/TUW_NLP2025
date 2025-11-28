# Qualitative Comparison: TF-IDF vs VerbatimRAG

| Query                                                                                       | Ground Truth   | TF-IDF Top-1     | VerbatimRAG Top-1   | Notes                                               |
|:--------------------------------------------------------------------------------------------|:---------------|:-----------------|:--------------------|:----------------------------------------------------|
| Are deep learning methods effective for crime forecasting compared to traditional models?   | 2509.20913v1   | 2509.20913v1 (✓) | 2509.20913v1 (✓)    |                                                     |
| Should I train separate models for different crime types, or combine them?                  | 2509.20913v1   | 2509.20913v1 (✓) | 2509.20913v1 (✓)    |                                                     |
| Which deep learning approaches work well for gamma/hadron separation?                       | 2510.05736v1   | 2510.05736v1 (✓) | 2510.05736v1 (✓)    |                                                     |
| What frameworks and optimization strategies were used to train DPCformer?                   | 2510.08662v1   | 2510.08662v1 (✓) | 2510.08662v1 (✓)    |                                                     |
| How do modern architectures perform on complex video tasks compared to older methods?       | 2510.09187v1   | 2510.09187v1 (✓) | 2510.09187v1 (✓)    |                                                     |
| How can I train models stably with limited computational resources?                         | 2510.12850v1   | 2510.13137v1 (✗) | 2509.20913v1 (✗)    | TF-IDF missed in top-5; VerbatimRAG missed in top-5 |
| Why is preprocessing important for Ethic-BERT's performance?                                | 2510.12850v1   | 2510.12850v1 (✓) | 2510.12850v1 (✓)    |                                                     |
| What are the main strengths of using an LSTM model for real-time sign language translation? | 2510.13137v1   | 2510.13137v1 (✓) | 2510.13137v1 (✓)    |                                                     |
| How does model selection affect responsiveness in real-time applications?                   | 2510.13137v1   | 2510.13137v1 (✓) | 2510.08662v1 (✗)    | VerbatimRAG correct at rank 4                       |

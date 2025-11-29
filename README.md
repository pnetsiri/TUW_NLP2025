# Topic 12: Retrieval-Augmented Generation for academic research

### 194.093 Natural Language Processing and Information Extraction, TU Wien, 2025 WS

#### Instructor: Gábor Recski

#### Students:

- Messeritsch Elina ([elinames](https://github.com/elinames))
- Netsiri Poj ([pnetsiri](https://github.com/pnetsiri))
- Selenge Tuvshin ([TuvshinSelenge](https://github.com/TuvshinSelenge))
- Vani Marvi ([marvi2424](https://github.com/marvi2424))

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for academic research. Our goal is to build a RAG system capable of answering a predefined set of questions based on a corpus of academic papers.

## Data

We downloaded 20 academic papers in the field of Deep Learning written in the English language (see the folder [paper/](paper/)). The PDF files were manually converted to text using an online tool, which can be found in the folder [paper_txt/](paper_txt/).

## Preprocessing

All preprocessing of the text files was done using the Jupyter notebook [code/text_processing.ipynb](code/text_processing.ipynb). The texts were first cleaned to remove common PDF hyphenation artifacts and then processed with the Stanza library to perform tokenization, part-of-speech tagging, lemmatization and dependency parsing. Each document was exported in the CoNLL-U format with one file per paper, which can be found in the folder [code/data/](code/data/).

The corpus is also stored in [corpus_json/corpus.json](corpus_json/corpus.json), where each entry contains the paper ID, title, and full cleaned text. This JSON structure is used as input for the retrieval systems.

## [Queries](queries_json/queries.json)

We use 20 different questions for evaluating the retrieval systems. Each query contains a question and, optionally, the correct paper ID, title, and the relevant passage.

We include two types of queries to get  a balanced view of overall retrieval quality and RAG accuracy:

1. Queries without a known correct paper

    These test how well the retrieval systems and RAG handle questions of different difficulty levels across a wide range of deep learning topics. Since there is no predefined answer, they show whether the system can still retrieve meaningful and relevant information.

2. Queries with a known correct paper and passage

    These allow us to explicitly evaluate accuracy: whether the system retrieves the correct paper, finds the relevant passage, and generates an answer close to the ground truth.

## Retrieval Systems and RAG

For milestone 2, we experimented with two types of retrieval systems:

1. [Rule-based TF-IDF system](milestone2/rule_based/)

    This system uses TF-IDF vectorization and cosine similarity to rank papers according to their relevance to a query.

2. [Machine-learning-based RAG system](milestone2/ML_based/)

    This system uses VerbatimRAG with SPLADE embeddings and a Milvus vector store to perform retrieval over document chunks.

The [evaluation script](milestone2/evaluation/evaluation.py) compares both systems by running all queries on the prepared corpus, computing key metrics (Precision, Recall, F1, Accuracy, Recall@k, MRR), and saving CSV results, qualitative comparisons, and performance plots. All outputs are in the [evaluation directory](milestone2/evaluation/), with full metrics, visualizations, and a detailed discussion in [Results.md](milestone2/evaluation/Results.md).

## Project Repository Structure

- [paper/](paper/): academic papers in original PDF format
- [paper_txt/](paper_txt/): extracted academic papers in TXT format
- [code/](code/): all scripts and notebooks used for text preprocessing
  - [data/](code/data/): preprocessed texts in CoNLL-U format
  - [requirements.txt](code/requirements.txt): libraries used
  - [text_processing.ipynb](code/text_processing.ipynb): notebook with all preprocessing steps
- [corpus_json/](corpus_json/): JSON file of the corpus
- [queries_json/](queries_json/): JSON file with evaluation questions
- [milestone2/](milestone2/): all files related to Milestone 2
  - [rule_based/](milestone2/rule_based/): TF-IDF–based retrieval system
    - [MileStone2_TF_IDF.ipynb](milestone2/rule_based/MileStone2_TF_IDF.ipynb): retrieval implementation with its preliminary evaluation
  - [ML_based/](milestone2/ML_based/): ML-based retrieval and RAG pipeline using VerbatimRAG
    - [ML_based_classification.ipynb](milestone2/ML_based/ML_based_classification.ipynb): RAG implementation
    - [ML_based_classification.py](milestone2/ML_based/ML_based_classification.py): RAG implementation in `.py` for importing functions needed for [rag_testing.ipynb](milestone2/ML_based/rag_testing.ipynb)
    - [milvus_final.db](milestone2/ML_based/milvus_final.db): Milvus database
    - [rag_testing.ipynb](milestone2/ML_based/rag_testing.ipynb): preliminary evaluation notebook
  - [evaluation/](milestone2/evaluation/): evaluation for milestone 2, comparing TF-IDF vs VerbatimRAG
    - [Results.md](milestone2/evaluation/Results.md): results of the comparison, summary and findings
    - [qualitative_comparison.md](milestone2/evaluation/qualitative_comparison.md): qualitative comparison
    - [evaluation.py](milestone2/evaluation/evaluation.py): evaluation script
    -  [confusion_matrix_metrics.csv](milestone2/evaluation/confusion_matrix_metrics.csv): TP/FP/FN/TN and derived metrics
    -  [comparison_metrics.csv](milestone2/evaluation/comparison_metrics.csv): retrieval metrics
    -  [results_tfidf.csv](milestone2/evaluation/results_tfidf.csv): detailed TF-IDF results
    -  [results_verbatimrag.csv](milestone2/evaluation/results_verbatimrag.csv): detailed VerbatimRAG results
    -  [comparison_chart.png](milestone2/evaluation/comparison_chart.png): visualization

## How to reproduce project

1. Clone the repository:

```bash
git clone https://github.com/pnetsiri/TUW_NLP2025.git
cd TUW_NLP2025
```

2. Install dependencies:

```bash
pip install -r code/requirements.txt
```

3. Download the spaCy large English model:

```bash
python -m spacy download en_core_web_lg
```

4. Dowload models:

```python
import nltk, stanza
nltk.download('punkt')
nltk.download('stopwords')
stanza.download('en')
```

5. Run the preprocessing notebook to generate the CoNLL-U files (or use the already provided `.conllu` files in [code/data/](code/data/)):

```bash
jupyter notebook code/text_processing.ipynb
```

6. Run Milestone 2 experiments:

- Rule-based system:

```bash
jupyter notebook milestone2/rule_based/MileStone2_TF_IDF.ipynb
```

- ML-based RAG system:

```bash
jupyter notebook milestone2/ML_based/ML_based_classification.ipynb
jupyter notebook milestone2/ML_based/rag_testing.ipynb
```

7. Reproduce the evaluation by running:

```bash
python milestone2/evaluation/evaluation.py
```

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
    - [MileStone2_TF_IDF.ipynb](milestone2/rule_based/MileStone2_TF_IDF.ipynb): retrieval implementation and its evaluation
  - [ML_based/](milestone2/ML_based/): ML-based retrieval and RAG pipeline using VerbatimRAG
    - [ML_based_classification.ipynb](milestone2/ML_based/ML_based_classification.ipynb): RAG implementation
    - [ML_based_classification.py](milestone2/ML_based/ML_based_classification.py): RAG implementation in `.py` for importing functions.
    - [milvus_final.db](milestone2/ML_based/milvus_final.db): Milvus database
    - [rag_testing.ipynb](milestone2/ML_based/rag_testing.ipynb): evaluation notebook

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

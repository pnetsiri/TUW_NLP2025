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

We downloaded 20 academic papers in the field of Deep Learning written in the English language. The PDF files were manually converted to text using an online tool.

## Preprocessing

All preprocessing of the text files was done using the Jupyter notebook [code/text_processing.ipynb](code/text_processing.ipynb). The NLTK library was used for tokenization and stopword removal, spaCy for lemmatization, and Stanza for exporting the preprocessed texts in the CoNLL-U format.

## Project Repository Structure

- [paper](paper/): academic papers in original PDF format
- [paper_txt](paper_txt/): extracted academic papers in TXT format
- [code](code/): all code of the project
  - [code/data](code/data/): preprocessed texts in CoNLL-U format
  - [code/requirements.txt](code/requirements.txt): libraries used
  - [code/text_processing.ipynb](code/text_processing.ipynb): Jupyter notebook with all preprocessing steps

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

3. Run the preprocessing Jupyter Notebook:

```bash
jupyter notebook code/text_processing.ipynb
```

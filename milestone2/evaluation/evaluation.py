"""
Milestone 2: RAG Evaluation Script

This script evaluates complete RAG systems (Retrieval + Generation):
- Keyword retriever (baseline)
- BM25 retriever (baseline)
- TF-IDF document-level retriever (baseline)
- TF-IDF chunk-level retriever (baseline)

Outputs:
- Quantitative metrics: retrieval (Recall@k, MRR, nDCG) + generation (BLEU, ROUGE)
- Answer quality metrics (Faithfulness, Relevancy, Context Precision)
- Qualitative examples with generated answers
- CSV results for each baseline
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_reference_answers(test_queries):
    """
    Extract reference answers from queries.

    Args:
        test_queries: List of query dictionaries with optional 'reference_answer' field

    Returns:
        Dictionary mapping questions to reference answers
    """
    return {
        q["question"]: q["reference_answer"]
        for q in test_queries
        if q.get("reference_answer")
    }


def get_doc_text(doc):
    """
    Extract text content from a document or chunk.

    Args:
        doc: Dictionary containing either 'text' or 'chunk' field

    Returns:
        Text content as string
    """
    return doc.get('text', doc.get('chunk', ''))


def chunk_text(text, chunk_size=220, overlap=40):
    """Split text into overlapping chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        if end >= n:
            break

        start = end - overlap

    return chunks


# ============================================================================
# RETRIEVAL BASELINES
# ============================================================================

def keyword_retrieve(query, corpus, k=5):
    """
    Keyword-based retrieval using simple token overlap.
    Returns top-k documents with highest keyword overlap.
    """
    q_tokens = set(query.lower().split())
    scores = []

    for doc in corpus:
        d_tokens = set(doc["text"].lower().split())
        overlap_score = len(q_tokens & d_tokens)
        scores.append(overlap_score)

    topk_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(topk_indices, start=1):
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "id": corpus[idx]["id"],
            "title": corpus[idx].get("title", ""),
            "text": corpus[idx]["text"]
        })

    return results


class SimpleBM25:
    """BM25 implementation for ranking documents."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.idf = {}
        self._calc_idf()

    def _calc_idf(self):
        """Calculate IDF for all terms."""
        df = {}
        for doc in self.corpus:
            tokens = set(doc.lower().split())
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        num_docs = len(self.corpus)
        for token, freq in df.items():
            self.idf[token] = np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        """Get BM25 scores for query against all documents."""
        query_tokens = query.lower().split()
        scores = []

        for doc, doc_len in zip(self.corpus, self.doc_lengths):
            doc_tokens = doc.lower().split()
            score = 0

            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = doc_tokens.count(token)
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        return scores


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def regex_extract_answer(text):
    """
    Extract answer using simple regex patterns.
    Looks for numbers, percentages, or returns first sentence.
    """
    # Try to find numbers or percentages
    match = re.findall(r"\b\d+\.?\d*%?\b", text)
    if match:
        return match[0]

    # Fall back to first sentence (max 150 chars)
    first_sentence = text.split(".")[0][:150]
    return first_sentence


def generate_answer_extractive(question, retrieved_docs):
    """
    Generate answer by extracting relevant text from retrieved documents.
    Uses keyword matching to find most relevant sentences.
    """
    if not retrieved_docs:
        return "No relevant information found."

    # Extract keywords from question
    stop_words = {'how', 'what', 'why', 'when', 'where', 'who', 'is', 'are',
                  'the', 'a', 'an', 'in', 'on', 'can', 'do', 'does', 'i', '?'}
    question_keywords = set(question.lower().split()) - stop_words

    # Collect sentences from top documents
    all_sentences = []
    for doc in retrieved_docs[:3]:
        text = get_doc_text(doc)
        sentences = text.replace('\n', ' ').split('. ')

        for sent in sentences:
            if len(sent.strip()) > 30:
                sent_lower = sent.lower()
                overlap = sum(1 for kw in question_keywords if kw in sent_lower)
                if overlap > 0:
                    all_sentences.append((overlap, sent))

    # Sort by keyword overlap
    all_sentences.sort(reverse=True, key=lambda x: x[0])

    if all_sentences:
        # Return top 2 sentences
        answer_parts = [sent for _, sent in all_sentences[:2]]
        return '. '.join(answer_parts) + '.'
    else:
        # Use regex extraction as fallback
        top_text = get_doc_text(retrieved_docs[0])
        return regex_extract_answer(top_text)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_bleu_score(reference, candidate):
    """Compute simplified BLEU score (unigram)."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not cand_tokens:
        return 0.0

    # Unigram precision
    matches = sum(1 for token in cand_tokens if token in ref_tokens)
    precision = matches / len(cand_tokens) if cand_tokens else 0

    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens)/len(cand_tokens))

    return bp * precision


def compute_rouge_scores(reference, candidate):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not cand_tokens or not ref_tokens:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    # ROUGE-1 (unigram overlap)
    ref_unigrams = set(ref_tokens)
    cand_unigrams = set(cand_tokens)
    overlap = len(ref_unigrams & cand_unigrams)
    rouge1_precision = overlap / len(cand_unigrams) if cand_unigrams else 0
    rouge1_recall = overlap / len(ref_unigrams) if ref_unigrams else 0
    rouge1_f1 = 2 * rouge1_precision * rouge1_recall / (rouge1_precision + rouge1_recall) if (rouge1_precision + rouge1_recall) > 0 else 0

    # ROUGE-2 (bigram overlap)
    def get_bigrams(tokens):
        return set(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))

    ref_bigrams = get_bigrams(ref_tokens)
    cand_bigrams = get_bigrams(cand_tokens)
    bigram_overlap = len(ref_bigrams & cand_bigrams)
    rouge2_precision = bigram_overlap / len(cand_bigrams) if cand_bigrams else 0
    rouge2_recall = bigram_overlap / len(ref_bigrams) if ref_bigrams else 0
    rouge2_f1 = 2 * rouge2_precision * rouge2_recall / (rouge2_precision + rouge2_recall) if (rouge2_precision + rouge2_recall) > 0 else 0

    # ROUGE-L (longest common subsequence)
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs_len = lcs_length(ref_tokens, cand_tokens)
    rougeL_precision = lcs_len / len(cand_tokens) if cand_tokens else 0
    rougeL_recall = lcs_len / len(ref_tokens) if ref_tokens else 0
    rougeL_f1 = 2 * rougeL_precision * rougeL_recall / (rougeL_precision + rougeL_recall) if (rougeL_precision + rougeL_recall) > 0 else 0

    return {
        'rouge1': rouge1_f1,
        'rouge2': rouge2_f1,
        'rougeL': rougeL_f1
    }


def recall_at_k(retrieved_ids, correct_id, k):
    """Check if correct document is in top-k retrieved documents."""
    if correct_id is None or correct_id == "None":
        return None
    return 1 if str(correct_id) in [str(id) for id in retrieved_ids[:k]] else 0


def mean_reciprocal_rank(retrieved_ids, correct_id):
    """Calculate reciprocal rank of the correct document."""
    if correct_id is None or correct_id == "None":
        return None

    correct_id_str = str(correct_id)
    retrieved_ids_str = [str(id) for id in retrieved_ids]

    if correct_id_str in retrieved_ids_str:
        rank = retrieved_ids_str.index(correct_id_str) + 1
        return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids, correct_id, k):
    """Calculate Discounted Cumulative Gain at k."""
    if correct_id is None or correct_id == "None":
        return None

    correct_id_str = str(correct_id)
    retrieved_ids_str = [str(id) for id in retrieved_ids[:k]]

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids_str):
        if doc_id == correct_id_str:
            dcg += 1.0 / np.log2(i + 2)
            break
    return dcg


def ndcg_at_k(retrieved_ids, correct_id, k):
    """Calculate Normalized Discounted Cumulative Gain at k."""
    dcg = dcg_at_k(retrieved_ids, correct_id, k)
    if dcg is None:
        return None

    # Ideal DCG (correct doc at position 0)
    idcg = 1.0 / np.log2(2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_faithfulness(answer, retrieved_contexts):
    """Measure if answer is faithful to retrieved context."""
    if not answer or not retrieved_contexts:
        return 0.0

    # Combine all contexts
    full_context = " ".join([get_doc_text(ctx) for ctx in retrieved_contexts])
    full_context_lower = full_context.lower()

    # Split answer into sentences
    answer_sentences = answer.split('. ')

    # Check how many answer sentences have support in context
    supported = 0
    for sent in answer_sentences:
        if not sent.strip():
            continue
        sent_tokens = set(sent.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'in', 'on'}
        if sent_tokens:
            overlap = sum(1 for token in sent_tokens if token in full_context_lower)
            if overlap / len(sent_tokens) > 0.5:
                supported += 1

    return supported / len(answer_sentences) if answer_sentences else 0.0


def compute_answer_relevancy(question, answer):
    """Measure if answer is relevant to the question."""
    if not answer or not question:
        return 0.0

    question_lower = question.lower()
    answer_lower = answer.lower()

    # Extract keywords from question
    stop_words = {'how', 'what', 'why', 'when', 'where', 'who', 'is', 'are',
                  'the', 'a', 'an', 'in', 'on', 'can', 'do', 'does', 'i', '?'}
    question_keywords = set(question_lower.split()) - stop_words

    if not question_keywords:
        return 0.5

    # Check overlap
    overlap = sum(1 for kw in question_keywords if kw in answer_lower)
    relevancy = overlap / len(question_keywords)

    return min(relevancy, 1.0)


def compute_context_precision(retrieved_contexts, correct_paper_id):
    """Measure precision of retrieved contexts."""
    if not correct_paper_id or not retrieved_contexts:
        return None

    correct_id_str = str(correct_paper_id)

    # Check how early the correct document appears
    for i, ctx in enumerate(retrieved_contexts):
        doc_id = ctx.get('id', ctx.get('doc_id', ''))
        if str(doc_id) == correct_id_str:
            return 1.0 / (i + 1)

    return 0.0


# ============================================================================
# METRIC AGGREGATION
# ============================================================================

def compute_retrieval_metrics(results_df):
    """Compute all retrieval metrics from results dataframe."""
    valid_results = results_df[results_df["correct_paper"].notna()].copy()

    if len(valid_results) == 0:
        return {}

    metrics = {}

    # Accuracy (top-1)
    metrics["Accuracy"] = valid_results["is_correct"].mean()

    # Recall@k
    for k in [1, 3, 5]:
        recall_k_values = []
        for _, row in valid_results.iterrows():
            r = recall_at_k(row["retrieved_ids"], row["correct_paper"], k)
            if r is not None:
                recall_k_values.append(r)
        metrics[f"Recall@{k}"] = np.mean(recall_k_values) if recall_k_values else 0.0

    # MRR
    mrr_values = []
    for _, row in valid_results.iterrows():
        mrr = mean_reciprocal_rank(row["retrieved_ids"], row["correct_paper"])
        if mrr is not None:
            mrr_values.append(mrr)
    metrics["MRR"] = np.mean(mrr_values) if mrr_values else 0.0

    # nDCG@k
    for k in [3, 5]:
        ndcg_values = []
        for _, row in valid_results.iterrows():
            ndcg = ndcg_at_k(row["retrieved_ids"], row["correct_paper"], k)
            if ndcg is not None:
                ndcg_values.append(ndcg)
        metrics[f"nDCG@{k}"] = np.mean(ndcg_values) if ndcg_values else 0.0

    # Precision, Recall, F1 (using is_correct which already compares predicted vs correct)
    y_pred = valid_results["is_correct"].astype(int)
    y_true = np.ones(len(y_pred), dtype=int)  # All should be correct (ideal case)

    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def compute_generation_metrics(results_df, reference_answers):
    """Compute BLEU and ROUGE metrics for generated answers."""
    valid_results = results_df[results_df["question"].isin(reference_answers.keys())].copy()

    if len(valid_results) == 0:
        return {}

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for _, row in valid_results.iterrows():
        question = row["question"]
        generated_answer = row.get("generated_answer", "")

        if question in reference_answers:
            reference = reference_answers[question]

            # BLEU
            bleu = compute_bleu_score(reference, generated_answer)
            bleu_scores.append(bleu)

            # ROUGE
            rouge = compute_rouge_scores(reference, generated_answer)
            rouge1_scores.append(rouge['rouge1'])
            rouge2_scores.append(rouge['rouge2'])
            rougeL_scores.append(rouge['rougeL'])

    return {
        "BLEU": np.mean(bleu_scores) if bleu_scores else 0.0,
        "ROUGE-1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "ROUGE-2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "ROUGE-L": np.mean(rougeL_scores) if rougeL_scores else 0.0
    }


def compute_quality_metrics(results_df):
    """Compute answer quality metrics."""
    valid_results = results_df.dropna(subset=["generated_answer"]).copy()

    if len(valid_results) == 0:
        return {}

    faithfulness_scores = []
    relevancy_scores = []
    context_precision_scores = []

    for _, row in valid_results.iterrows():
        question = row["question"]
        answer = row["generated_answer"]
        retrieved_contexts = row.get("retrieved_contexts", [])
        correct_paper = row.get("correct_paper")

        # Faithfulness
        faith = compute_faithfulness(answer, retrieved_contexts)
        faithfulness_scores.append(faith)

        # Answer Relevancy
        relev = compute_answer_relevancy(question, answer)
        relevancy_scores.append(relev)

        # Context Precision
        if correct_paper:
            ctx_prec = compute_context_precision(retrieved_contexts, correct_paper)
            if ctx_prec is not None:
                context_precision_scores.append(ctx_prec)

    return {
        "Faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "Answer_Relevancy": np.mean(relevancy_scores) if relevancy_scores else 0.0,
        "Context_Precision": np.mean(context_precision_scores) if context_precision_scores else 0.0
    }


# ============================================================================
# EVALUATION LOOPS
# ============================================================================

def evaluate_baseline(retrieve_fn, test_queries, baseline_name):
    """
    Generic evaluation loop for any retrieval baseline.

    Args:
        retrieve_fn: Function that takes (query, k) and returns list of retrieved docs
        test_queries: List of test query dictionaries
        baseline_name: Name of the baseline for logging

    Returns:
        DataFrame with results
    """
    results = []

    for entry in test_queries:
        question = entry["question"]
        correct_paper = entry["correct_paper_id"]

        # Retrieve documents
        retrieved = retrieve_fn(question, k=5)

        retrieved_ids = [r["id"] for r in retrieved]
        top_pred = retrieved[0]["id"] if retrieved else None

        # Generate answer
        generated_answer = generate_answer_extractive(question, retrieved)

        results.append({
            "question": question,
            "predicted_paper": top_pred,
            "correct_paper": str(correct_paper) if correct_paper else None,
            "is_correct": (str(correct_paper) == str(top_pred) if correct_paper else None),
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [r["score"] for r in retrieved],
            "retrieved_titles": [r.get("title", "") for r in retrieved],
            "retrieved_contexts": retrieved,
            "generated_answer": generated_answer
        })

    df = pd.DataFrame(results)
    print(f"{baseline_name}: Evaluated {len(df)} queries")

    return df


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("=" * 80)
    print("MILESTONE 2: RAG EVALUATION (Retrieval + Generation)")
    print("=" * 80)

    # Load data
    base_path = Path(__file__).parent.parent.parent
    corpus_path = base_path / "corpus_json" / "corpus.json"
    query_path = base_path / "queries_json" / "queries.json"

    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    with query_path.open("r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"\nLoaded {len(corpus)} documents")
    print(f"Loaded {len(test_queries)} test queries")

    queries_with_gt = [q for q in test_queries if q["correct_paper_id"] is not None]
    print(f"Queries with ground truth: {len(queries_with_gt)}")

    # Load reference answers from queries
    reference_answers = get_reference_answers(test_queries)
    print(f"Reference answers: {len(reference_answers)}")

    # Prepare corpus data
    doc_ids = [doc["id"] for doc in corpus]
    doc_titles = [doc.get("title", "") for doc in corpus]
    doc_texts = [doc["text"] for doc in corpus]

    # ========================================================================
    # BASELINE 1: KEYWORD RETRIEVAL
    # ========================================================================

    print("\n" + "-" * 80)
    print("BASELINE 1: KEYWORD RETRIEVAL")
    print("-" * 80)

    def retrieve_keyword(query, k=5):
        return keyword_retrieve(query, corpus, k)

    df_keyword = evaluate_baseline(retrieve_keyword, test_queries, "Keyword")

    # ========================================================================
    # BASELINE 2: BM25 RETRIEVAL
    # ========================================================================

    print("\n" + "-" * 80)
    print("BASELINE 2: BM25 RETRIEVAL")
    print("-" * 80)

    bm25 = SimpleBM25(doc_texts)

    def retrieve_bm25(query, k=5):
        if not query.strip():
            return []
        scores = bm25.get_scores(query)
        topk_idx = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "id": doc_ids[idx],
                "title": doc_titles[idx],
                "text": doc_texts[idx]
            })
        return results

    df_bm25 = evaluate_baseline(retrieve_bm25, test_queries, "BM25")

    # ========================================================================
    # BASELINE 3: TF-IDF DOCUMENT-LEVEL
    # ========================================================================

    print("\n" + "-" * 80)
    print("BASELINE 3: TF-IDF DOCUMENT-LEVEL")
    print("-" * 80)

    vectorizer_doc = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1
    )
    tfidf_matrix_doc = vectorizer_doc.fit_transform(doc_texts)
    print(f"TF-IDF matrix: {tfidf_matrix_doc.shape}")

    def retrieve_tfidf_doc(query, k=5):
        if not query.strip():
            return []
        q_vec = vectorizer_doc.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix_doc)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            results.append({
                "rank": rank,
                "score": float(sims[idx]),
                "id": doc_ids[idx],
                "title": doc_titles[idx],
                "text": doc_texts[idx]
            })
        return results

    df_tfidf_doc = evaluate_baseline(retrieve_tfidf_doc, test_queries, "TF-IDF-Doc")

    # ========================================================================
    # BASELINE 4: TF-IDF CHUNK-LEVEL
    # ========================================================================

    print("\n" + "-" * 80)
    print("BASELINE 4: TF-IDF CHUNK-LEVEL")
    print("-" * 80)

    # Create chunks
    passage_texts = []
    passage_meta = []

    for doc in corpus:
        doc_id = doc["id"]
        title = doc.get("title", "")
        text = doc["text"]

        chunks = chunk_text(text, chunk_size=220, overlap=40)

        for i, chunk in enumerate(chunks):
            passage_texts.append(chunk)
            passage_meta.append({
                "id": doc_id,
                "title": title,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk
            })

    print(f"Created {len(passage_texts)} chunks from {len(corpus)} documents")

    vectorizer_chunk = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1
    )
    tfidf_matrix_chunk = vectorizer_chunk.fit_transform(passage_texts)
    print(f"TF-IDF matrix: {tfidf_matrix_chunk.shape}")

    def retrieve_tfidf_chunk(query, k=5):
        if not query.strip():
            return []
        q_vec = vectorizer_chunk.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix_chunk)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            meta = passage_meta[idx]
            results.append({
                "rank": rank,
                "score": float(sims[idx]),
                "id": meta["id"],
                "title": meta["title"],
                "text": meta["text"]
            })
        return results

    df_tfidf_chunk = evaluate_baseline(retrieve_tfidf_chunk, test_queries, "TF-IDF-Chunk")

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    # Retrieval metrics
    metrics_keyword = compute_retrieval_metrics(df_keyword)
    metrics_bm25 = compute_retrieval_metrics(df_bm25)
    metrics_tfidf_doc = compute_retrieval_metrics(df_tfidf_doc)
    metrics_tfidf_chunk = compute_retrieval_metrics(df_tfidf_chunk)

    # Generation metrics
    gen_keyword = compute_generation_metrics(df_keyword, reference_answers)
    gen_bm25 = compute_generation_metrics(df_bm25, reference_answers)
    gen_tfidf_doc = compute_generation_metrics(df_tfidf_doc, reference_answers)
    gen_tfidf_chunk = compute_generation_metrics(df_tfidf_chunk, reference_answers)

    # Quality metrics
    qual_keyword = compute_quality_metrics(df_keyword)
    qual_bm25 = compute_quality_metrics(df_bm25)
    qual_tfidf_doc = compute_quality_metrics(df_tfidf_doc)
    qual_tfidf_chunk = compute_quality_metrics(df_tfidf_chunk)

    # Combine metrics
    all_metrics = {
        "Keyword": {**metrics_keyword, **gen_keyword, **qual_keyword},
        "BM25": {**metrics_bm25, **gen_bm25, **qual_bm25},
        "TF-IDF (Document)": {**metrics_tfidf_doc, **gen_tfidf_doc, **qual_tfidf_doc},
        "TF-IDF (Chunk)": {**metrics_tfidf_chunk, **gen_tfidf_chunk, **qual_tfidf_chunk}
    }

    # Create comparison table
    metric_names = [
        "Accuracy", "Precision", "Recall", "F1",
        "Recall@1", "Recall@3", "Recall@5",
        "MRR", "nDCG@3", "nDCG@5",
        "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "Faithfulness", "Answer_Relevancy", "Context_Precision"
    ]

    comparison_data = {"Baseline": list(all_metrics.keys())}
    for metric in metric_names:
        comparison_data[metric] = [all_metrics[baseline].get(metric, 0) for baseline in all_metrics.keys()]

    df_comparison = pd.DataFrame(comparison_data)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_dir = Path(__file__).parent
    df_comparison.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    df_keyword.to_csv(output_dir / "results_keyword.csv", index=False)
    df_bm25.to_csv(output_dir / "results_bm25_rag.csv", index=False)
    df_tfidf_doc.to_csv(output_dir / "results_tfidf_document_rag.csv", index=False)
    df_tfidf_chunk.to_csv(output_dir / "results_tfidf_chunk_rag.csv", index=False)

    print("\nResults saved:")
    print(f"  {output_dir / 'evaluation_metrics.csv'}")
    print(f"  {output_dir / 'results_keyword.csv'}")
    print(f"  {output_dir / 'results_bm25_rag.csv'}")
    print(f"  {output_dir / 'results_tfidf_document_rag.csv'}")
    print(f"  {output_dir / 'results_tfidf_chunk_rag.csv'}")

    # ========================================================================
    # QUALITATIVE EXAMPLES
    # ========================================================================

    print("\n" + "=" * 80)
    print("QUALITATIVE EXAMPLES")
    print("=" * 80)

    for i in range(min(3, len(queries_with_gt))):
        query = queries_with_gt[i]
        question = query["question"]
        correct_paper = query["correct_paper_id"]
        correct_title = query.get("correct_paper_title", "")

        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"Query: {question}")
        print(f"Ground Truth: [{correct_paper}] {correct_title}")

        if question in reference_answers:
            print(f"Reference Answer: {reference_answers[question][:150]}...")

        # Show results from each baseline
        idx = test_queries.index(query)

        print(f"\n{'-'*80}")
        print("Keyword Retrieval")
        print(f"{'-'*80}")
        row = df_keyword.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Answer: {row['generated_answer'][:150]}...")

        print(f"\n{'-'*80}")
        print("BM25 Retrieval")
        print(f"{'-'*80}")
        row = df_bm25.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Answer: {row['generated_answer'][:150]}...")

        print(f"\n{'-'*80}")
        print("TF-IDF Document")
        print(f"{'-'*80}")
        row = df_tfidf_doc.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Answer: {row['generated_answer'][:150]}...")

        print(f"\n{'-'*80}")
        print("TF-IDF Chunk")
        print(f"{'-'*80}")
        row = df_tfidf_chunk.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Answer: {row['generated_answer'][:150]}...")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Format for display
    df_display = df_comparison.copy()
    for col in df_display.columns:
        if col != "Baseline":
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.4f}")

    print("\n" + df_display.to_string(index=False))

    for baseline_name, metrics in all_metrics.items():
        print(f"\n{baseline_name}:")
        print(f"  Retrieval: Accuracy={metrics.get('Accuracy', 0)*100:.2f}%, Recall@3={metrics.get('Recall@3', 0)*100:.2f}%, MRR={metrics.get('MRR', 0):.4f}")
        print(f"  Generation: BLEU={metrics.get('BLEU', 0):.4f}, ROUGE-1={metrics.get('ROUGE-1', 0):.4f}")
        print(f"  Quality: Faithfulness={metrics.get('Faithfulness', 0):.4f}, Relevancy={metrics.get('Answer_Relevancy', 0):.4f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return df_comparison, df_keyword, df_bm25, df_tfidf_doc, df_tfidf_chunk


if __name__ == "__main__":
    main()

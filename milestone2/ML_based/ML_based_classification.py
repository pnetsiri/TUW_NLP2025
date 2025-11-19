#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%pip install verbatim-rag


# In[2]:


import json
import pandas as pd
import os
from pathlib import Path
from verbatim_rag.document import Document, Chunk, ProcessedChunk, DocumentType, ChunkType
from verbatim_rag.ingestion import DocumentProcessor 
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag import VerbatimIndex
from verbatim_rag.embedding_providers import SpladeProvider
from collections import defaultdict


# In[3]:


documents_for_index = [] 

corpus_path = Path("../../corpus_json/corpus.json")
with corpus_path.open("r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loading {len(corpus)} papers...")


# In[4]:


# checking the corpus
df = pd.DataFrame(corpus)
df.head()


# ### Chunking

# In[5]:


# replicates the private method '_add_document_metadata' from the repo
def create_enhanced_content(text, doc):
    parts = [text, "", "---"]
    parts.append(f"Document: {doc.title or 'Unknown'}")
    parts.append(f"Source: {doc.source or 'Unknown'}")
    for key, value in doc.metadata.items():
         parts.append(f"{key}: {value}")
    return "\n".join(parts)


# In[6]:


# We initialize the processor and use its 'chunker_provider'
processor = DocumentProcessor()

for paper in corpus:
    # Create the shell Document object
    doc_obj = Document(
        title=paper['title'],
        source="json_corpus", 
        content_type=DocumentType.TXT, 
        raw_content=paper['text'],
        metadata={
            "id": paper['id'],
            "title": paper['title']
        }
    )
    # Manually Chunk the text using the processor's tool
    # This breaks the text into semantic pieces
    chunk_tuples = processor.chunker_provider.chunk(paper['text'])

    # Build Chunk objects
    for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):

        # Create the footer/header info
        enhanced_content = create_enhanced_content(struct_enhanced, doc_obj)

        # Create the Basic Chunk
        doc_chunk = Chunk(
            document_id=doc_obj.id,
            content=raw_text,
            chunk_number=i,
            chunk_type=ChunkType.PARAGRAPH,
        )

        # Create the Processed Chunk (The part that gets embedded)
        processed_chunk = ProcessedChunk(
            chunk_id=doc_chunk.id,
            enhanced_content=enhanced_content,
        )

        # Link them
        doc_chunk.add_processed_chunk(processed_chunk)
        doc_obj.add_chunk(doc_chunk)

    documents_for_index.append(doc_obj)


# ### Building the Index

# In[7]:


DB_FILE = "./milvus_final.db"

db_exists = os.path.exists(DB_FILE)

# Setup Store
# we explicitly tell the store we are using Sparse only to save memory
store = LocalMilvusStore(DB_FILE, enable_sparse=True, enable_dense=False)

# we use a standard SPLADE model that works well on CPUs
sparse_embedder = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cpu"
)
index = VerbatimIndex(vector_store=store, sparse_provider=sparse_embedder)

if db_exists:
    # The file exists on disk, so we check if Milvus can read it
    print("Database file found.")
    try:
        # We use a valid filter 'id != ""' instead of empty string
        res = store.client.query(store.collection_name, filter='id != ""', limit=1)
        if len(res) > 0:
            print("Index is already populated. SKIPPING ingestion.")
        else:
            print("Database exists but seems empty. Adding documents...")
            index.add_documents(documents_for_index)
    except Exception as e:
        print(f"Database seems corrupted: {e}")
        print("deleting and rebuilding...")
        store.client.drop_collection(store.collection_name)
        index.add_documents(documents_for_index)
else:
    print("New Database. Indexing documents...")
    index.add_documents(documents_for_index)


# ### Query

# In[ ]:


def find_best_paper(query_text, top_k=5):
    print(f"Querying: '{query_text}'")

    results = index.query(query_text, k=top_k)

    if not results:
        print("No matches found.")
        return None

    # Dictionary to accumulate REAL scores
    # {'Paper Title': 14.53}
    paper_scores = defaultdict(float)

    print(f"\n--- Top {top_k} Chunks & Actual Similarity Scores ---")

    for i, res in enumerate(results):
        # 1. Get Metadata (Title)
        meta = getattr(res, 'metadata', {}) or {}
        if not meta and hasattr(res, 'get'): meta = res.get('metadata', {})
        title = meta.get('title', meta.get('id', 'Unknown'))
        id = meta.get('id', 'Unknown')

        # 2. EXTRACT THE REAL SCORE
        # We try common attribute names used by Milvus wrappers
        score = getattr(res, 'score', None)

        # If .score is missing, sometimes it is called .distance
        if score is None:
            score = getattr(res, 'distance', 0.0)

        # 3. Add to Total
        paper_scores[id] += score

        # 4. Print Result
        # We print the score to 4 decimal places
        snippet = getattr(res, 'text', getattr(res, 'content', ''))[:40].replace('\n', '')
        print(f"Rank {i+1}: Score {score:.4f} | Paper: {id} [{title[:20]}...] | Text: {snippet}...")

    # 5. Winner
    sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    winner, total_score = sorted_papers[0]

    print("\n--- Classification Result ---")
    print(f"Predicted Paper: {winner}")
    print(f"Total Similarity Score: {total_score:.4f}")

    return winner # id of the paper


# ### Enter Query

# In[8]:


query = "How can we detect sarcasm using deep learning?"

#predicted_paper = find_best_paper(query)


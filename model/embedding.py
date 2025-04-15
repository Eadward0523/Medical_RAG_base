import requests
import os
from tqdm.auto import tqdm
import zipfile
import json
import pandas as pd
import numpy as np
from data_loader import load_jsonl_to_dataframe
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

def embed_jsonl_for_rag(jsonl_path, db_path, text_field="text", 
                         chunk_size=512, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Process a JSONL file and store embeddings in a vector database for RAG.
    
    Args:
        jsonl_path (str): Path to JSONL file
        db_path (str): Path to store the vector database
        text_field (str): Field in the JSONL containing text to embed
        chunk_size (int): Size of text chunks for embedding
        embedding_model_name (str): Name of sentence-transformers model
    """
    # Load the dataset
    medical_df = load_jsonl_to_dataframe()
    if medical_df is not None:
        print(f"Dataset contains {len(medical_df)} rows and {len(medical_df.columns)} columns")
        print("\nFirst few rows:")
        print(medical_df.head())
        
        print("\nColumn information:")
        for col in medical_df.columns:
            print(f"- {col}") 
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="jsonl_embeddings",
        embedding_function=embedding_function
    )

    batch_size = 10
    for i in tqdm(range(0, len(medical_df['context']), batch_size)):
        batch_contexts = medical_df['context'][i:i+batch_size]
        batch_ids = medical_df['id'][i:i+batch_size]
        batch_metadata = medical_df['metadata'][i:i+batch_size]
        
        collection.add(
            documents=batch_contexts,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    
    print(f"Successfully embedded {len(medical_df['context'])} documents to vector database at {db_path}")
    return collection

    
    
if __name__ == "__main__":
    collection = embed_jsonl_for_rag(
        jsonl_path='data/test.jsonl',
        db_path="./vector_db",
        text_field="context"
    )
    
    results = collection.query(
        query_texts=["sample query"],
        n_results=5
    )
    print("Query results:", results)


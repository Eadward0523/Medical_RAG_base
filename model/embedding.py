import os
import tqdm
import zipfile
import json
import pandas as pd
import numpy as np
import random
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
    for i in tqdm.tqdm(range(0, len(medical_df['context']), batch_size)):
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

def get_random_samples_from_db(db_path, collection_name="jsonl_embeddings", n_samples=5):
    """
    Load random samples from a ChromaDB vector database.
    
    Args:
        db_path (str): Path to the vector database
        collection_name (str): Name of the collection to query
        n_samples (int): Number of random samples to retrieve
        
    Returns:
        dict: Dictionary containing the random samples with their documents, metadatas, and ids
    """
    try:
        # Connect to the database
        client = chromadb.PersistentClient(path=db_path)
        
        # Get the collection
        collection = client.get_collection(name=collection_name)
        
        # Get all IDs in the collection
        all_ids = collection.get()["ids"]
        
        if not all_ids:
            print(f"No documents found in collection '{collection_name}'")
            return None
        
        # Select random IDs
        if n_samples > len(all_ids):
            n_samples = len(all_ids)
            print(f"Warning: Requested more samples than available. Returning all {n_samples} samples.")
        
        random_ids = random.sample(all_ids, n_samples)
        
        # Get the documents for the random IDs
        results = collection.get(ids=random_ids)
        
        print(f"Successfully retrieved {len(results['ids'])} random samples from the database")
        
        return results
    
    except Exception as e:
        print(f"Error retrieving random samples: {str(e)}")
        return None
    
    
if __name__ == "__main__":
    # Create embeddings and store in database
    collection = embed_jsonl_for_rag(
        jsonl_path='data/test.jsonl',
        db_path="./vector_db",
        text_field="context"
    )
    
    # Query the database
    results = collection.query(
        query_texts=["sample query"],
        n_results=5
    )
    print("Query results:", results)
    
    # Get random samples from the database
    random_samples = get_random_samples_from_db(
        db_path="./vector_db",
        n_samples=3
    )
    
    if random_samples:
        print("\nRandom samples from the database:")
        for i, (doc, metadata, doc_id) in enumerate(zip(random_samples['documents'], 
                                                        random_samples['metadatas'], 
                                                        random_samples['ids'])):
            print(f"\nSample {i+1} (ID: {doc_id}):")
            print(f"Document: {doc[:100]}...")  # Print first 100 characters
            print(f"Metadata: {metadata}")


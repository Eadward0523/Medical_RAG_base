import os
import torch
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
from time import perf_counter as timer


def embed_jsonl(jsonl_path, db_path, text_field="text", 
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
        model_name=embedding_model_name,
    )
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="jsonl_embeddings",
        embedding_function=embedding_function
    )

    batch_size = 10
    for i in tqdm.tqdm(range(0, len(medical_df['context']), batch_size)):
        # Get the batch slice first
        batch_df = medical_df.iloc[i:i+batch_size]
        
        batch_contexts = batch_df['context'].tolist()
        batch_ids = batch_df['qid'].astype(str).tolist()
        batch_metadata = [
            {
                'diagnosis': row['diagnosis'],
                'diagnosis_code': row['diagnosis_code']
            }
            for _, row in batch_df.iterrows()
        ]
        
        # Debug print to verify lengths
        print(f"Batch {i//batch_size + 1}: contexts={len(batch_contexts)}, ids={len(batch_ids)}, metadata={len(batch_metadata)}")
        
        collection.add(
            documents=batch_contexts,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    
    print(f"Successfully embedded {len(medical_df['context'])} documents to vector database at {db_path}")
    return collection

def get_random_samples_from_db(db_path, collection_name="jsonl_embeddings", n_samples=5):
    """
    Retrieve relevant samples from a ChromaDB vector database.
    
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
    

def retrieve_relevant_resources(query: str, collection, model, k: int = 5):
    """
    Retrieves the top k most relevant resources based on cosine similarity.
    
    Args:
        query (str): The input query string
        collection: The ChromaDB collection containing embeddings
        model: The embedding model to use
        k (int): Number of top results to return (default: 5)
    
    Returns:
        tuple: (scores, indices) of the top k most similar documents
    """
    # Embed the query
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Get results from collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    
    # Extract scores and indices
    scores = results['distances'][0]  # First element since we only have one query
    indices = results['ids'][0]
    
    return scores, indices

    
def print_top_results_and_scores(query: str, collection, model, k: int = 5):
        """
        Retrieves and prints the top k most relevant resources with their scores and contexts.
        
        Args:
            query (str): The input query string
            collection: The ChromaDB collection containing embeddings
            model: The embedding model to use
            k (int): Number of top results to return (default: 5)
        """
        scores, indices = retrieve_relevant_resources(query, collection, model, k)
        
        results = collection.get(ids=indices)
        for i, (doc, metadata, doc_id) in enumerate(zip(results['documents'], 
                                                        results['metadatas'], 
                                                        results['ids'])):
            print(f"\nQuery {i+1} (ID: {doc_id}):")
            print(f"Context: {doc}...")  # Print first 100 characters
            print(f"Diagnosis: {metadata}")
    
    
if __name__ == "__main__":
    # Create embeddings and store in database
    db_path="./vector_db"

    collection = embed_jsonl(
        jsonl_path='data/test.jsonl',
        db_path=db_path,
        text_field="context"
    )
    
    # Load the model (e.g., SentenceTransformer)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example query
    query = "autism"
    print_top_results_and_scores(query, collection, model, k=5)
    
    '''# Query the database
    results = collection.query(
        query_texts=["sample query"],
        n_results=5
    )
    print("Query results:", results)'''
    
    ''' Get random samples from the database
    random_samples = get_random_samples_from_db(
        db_path=db_path,
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
    '''



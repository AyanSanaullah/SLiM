from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import uvicorn
import os
import numpy as np
from typing import List, Dict
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI(title="Semantic Comparison API")

# Load the models once at startup
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ComparisonRequest(BaseModel):
    sentence1: str
    sentence2: str

class WordAnalysis(BaseModel):
    word: str
    pos_tag: str
    semantic_similarity: float
    wordnet_similarity: float
    embedding_similarity: float
    overall_score: float
    explanation: str

class ComparisonResponse(BaseModel):
    similarity: float
    sentence1: str
    sentence2: str
    word_analysis: List[WordAnalysis]

def semantic_similarity(a: str, b: str) -> float:
    """Calculate semantic similarity between two sentences"""
    # Encode -> L2-normalized embeddings
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    # Cosine similarity in [-1,1]; with normalized vecs it's the dot product
    sim = torch.matmul(emb[0], emb[1]).item()
    # Map from [-1,1] to [0,1] if you prefer
    return (sim + 1) / 2

def get_wordnet_similarity(word1: str, word2: str) -> float:
    """Calculate WordNet-based semantic similarity between two words"""
    try:
        synsets1 = wordnet.synsets(word1)
        synsets2 = wordnet.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        max_similarity = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                similarity = syn1.wup_similarity(syn2)
                if similarity is not None:
                    max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    except:
        return 0.0

def analyze_words_importance(sentence1: str, sentence2: str) -> List[WordAnalysis]:
    """Analyze word-by-word semantic importance by comparing words between the two sentences"""
    # Process sentences with spaCy
    doc1 = nlp(sentence1.lower())
    doc2 = nlp(sentence2.lower())
    
    # Get meaningful words (nouns, verbs, adjectives, adverbs) from each sentence
    words1 = [token.lemma_ for token in doc1 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop and not token.is_punct]
    words2 = [token.lemma_ for token in doc2 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop and not token.is_punct]
    
    # Get POS tags for words
    word_pos_map = {}
    for token in doc1:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop and not token.is_punct:
            word_pos_map[token.lemma_] = token.pos_
    for token in doc2:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop and not token.is_punct:
            word_pos_map[token.lemma_] = token.pos_
    
    word_analyses = []
    
    # Compare each word from sentence1 with words from sentence2
    for word1 in set(words1):
        best_match = ""
        max_embedding_sim = 0.0
        max_wordnet_sim = 0.0
        
        # Check for exact match first
        if word1 in words2:
            max_embedding_sim = 1.0
            max_wordnet_sim = 1.0
            best_match = word1
        else:
            # Find best semantic match in sentence2
            for word2 in set(words2):
                # Calculate embedding similarity
                word1_emb = model.encode([word1], convert_to_tensor=True, normalize_embeddings=True)
                word2_emb = model.encode([word2], convert_to_tensor=True, normalize_embeddings=True)
                embedding_sim = torch.matmul(word1_emb[0], word2_emb[0]).item()
                embedding_sim = (embedding_sim + 1) / 2  # Normalize to [0,1]
                
                # Calculate WordNet similarity
                wordnet_sim = get_wordnet_similarity(word1, word2)
                
                # Keep track of best match
                combined_score = embedding_sim * 0.6 + wordnet_sim * 0.4
                if combined_score > (max_embedding_sim * 0.6 + max_wordnet_sim * 0.4):
                    max_embedding_sim = embedding_sim
                    max_wordnet_sim = wordnet_sim
                    best_match = word2
        
        # Calculate semantic similarity
        semantic_sim = max_embedding_sim * 0.7 + max_wordnet_sim * 0.3
        
        # Calculate overall score
        overall_score = semantic_sim
        
        # Generate explanation
        if word1 == best_match:
            explanation = f"Exact match: '{word1}' appears in both sentences"
        elif best_match:
            explanation = f"'{word1}' (sentence 1) ↔ '{best_match}' (sentence 2) - Similarity: {semantic_sim:.2f}"
        else:
            explanation = f"'{word1}' has no semantic match in sentence 2"
        
        word_analyses.append(WordAnalysis(
            word=word1,
            pos_tag=word_pos_map.get(word1, "UNKNOWN"),
            semantic_similarity=semantic_sim,
            wordnet_similarity=max_wordnet_sim,
            embedding_similarity=max_embedding_sim,
            overall_score=overall_score,
            explanation=explanation
        ))
    
    # Also compare words from sentence2 that don't have matches in sentence1
    for word2 in set(words2):
        if word2 not in [analysis.word for analysis in word_analyses]:
            best_match = ""
            max_embedding_sim = 0.0
            max_wordnet_sim = 0.0
            
            # Find best semantic match in sentence1
            for word1 in set(words1):
                # Calculate embedding similarity
                word1_emb = model.encode([word1], convert_to_tensor=True, normalize_embeddings=True)
                word2_emb = model.encode([word2], convert_to_tensor=True, normalize_embeddings=True)
                embedding_sim = torch.matmul(word1_emb[0], word2_emb[0]).item()
                embedding_sim = (embedding_sim + 1) / 2  # Normalize to [0,1]
                
                # Calculate WordNet similarity
                wordnet_sim = get_wordnet_similarity(word1, word2)
                
                # Keep track of best match
                combined_score = embedding_sim * 0.6 + wordnet_sim * 0.4
                if combined_score > (max_embedding_sim * 0.6 + max_wordnet_sim * 0.4):
                    max_embedding_sim = embedding_sim
                    max_wordnet_sim = wordnet_sim
                    best_match = word1
            
            # Calculate semantic similarity
            semantic_sim = max_embedding_sim * 0.7 + max_wordnet_sim * 0.3
            
            # Calculate overall score
            overall_score = semantic_sim
            
            # Generate explanation
            if best_match:
                explanation = f"'{word2}' (sentence 2) ↔ '{best_match}' (sentence 1) - Similarity: {semantic_sim:.2f}"
            else:
                explanation = f"'{word2}' has no semantic match in sentence 1"
            
            word_analyses.append(WordAnalysis(
                word=word2,
                pos_tag=word_pos_map.get(word2, "UNKNOWN"),
                semantic_similarity=semantic_sim,
                wordnet_similarity=max_wordnet_sim,
                embedding_similarity=max_embedding_sim,
                overall_score=overall_score,
                explanation=explanation
            ))
    
    # Sort by overall score
    word_analyses.sort(key=lambda x: x.overall_score, reverse=True)
    
    return word_analyses[:15]  # Return top 15 most semantically important words

@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    return FileResponse('index.html')

@app.post("/compare", response_model=ComparisonResponse)
async def compare_sentences(request: ComparisonRequest):
    """Compare two sentences semantically"""
    try:
        if not request.sentence1.strip() or not request.sentence2.strip():
            raise HTTPException(status_code=400, detail="Both sentences must be provided")
        
        similarity = semantic_similarity(request.sentence1, request.sentence2)
        word_analysis = analyze_words_importance(request.sentence1, request.sentence2)
        
        return ComparisonResponse(
            similarity=similarity,
            sentence1=request.sentence1,
            sentence2=request.sentence2,
            word_analysis=word_analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing comparison: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "sentence-transformers/all-MiniLM-L6-v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

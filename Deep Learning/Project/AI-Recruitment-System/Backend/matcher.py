# backend/matcher.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from typing import Tuple, List

EMBED_MODEL_NAME = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers/all-mpnet-base-v2")

embedder = SentenceTransformer(EMBED_MODEL_NAME)

def tfidf_similarity(doc1: str, doc2: str) -> float:
    # Use n-grams and better preprocessing for better matching
    vect = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Include bigrams
        max_features=5000,
        lowercase=True,
        strip_accents='unicode'
    )
    mats = vect.fit_transform([doc1, doc2])
    sim = cosine_similarity(mats[0:1], mats[1:2])[0][0]
    return float(sim)

def embed_texts(texts: List[str]):
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def semantic_similarity(doc1: str, doc2: str) -> float:
    e1 = embed_texts([doc1])[0]
    e2 = embed_texts([doc2])[0]
    # cosine:
    num = np.dot(e1, e2)
    den = np.linalg.norm(e1)*np.linalg.norm(e2)
    return float(num/den) if den != 0 else 0.0

def keyword_similarity(resume_text: str, job_description: str) -> float:
    """Calculate keyword-based similarity for better ATS scoring"""
    import re
    
    # Extract technical keywords from job description
    job_keywords = set()
    
    # Common technical terms
    tech_terms = [
        'python', 'java', 'javascript', 'react', 'django', 'flask', 'node.js',
        'sql', 'mongodb', 'git', 'docker', 'kubernetes', 'aws', 'azure',
        'machine learning', 'data science', 'statistics', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'product management', 'agile', 'scrum', 'ui/ux', 'figma', 'adobe',
        'devops', 'ci/cd', 'linux', 'shell scripting', 'cloud computing'
    ]
    
    # Extract keywords from job description
    job_lower = job_description.lower()
    for term in tech_terms:
        if term in job_lower:
            job_keywords.add(term)
    
    # Extract keywords from resume
    resume_keywords = set()
    resume_lower = resume_text.lower()
    for term in tech_terms:
        if term in resume_lower:
            resume_keywords.add(term)
    
    # Calculate Jaccard similarity
    if len(job_keywords) == 0:
        return 0.0
    
    intersection = len(job_keywords.intersection(resume_keywords))
    union = len(job_keywords.union(resume_keywords))
    
    return float(intersection / union) if union > 0 else 0.0

def calculate_ats_score(resume_text: str, job_description: str) -> dict:
    """Calculate comprehensive ATS score with multiple metrics"""
    # Calculate individual similarities
    tfidf_sim = tfidf_similarity(resume_text, job_description)
    semantic_sim = semantic_similarity(resume_text, job_description)
    keyword_sim = keyword_similarity(resume_text, job_description)
    
    # Calculate bonus points for comprehensive resumes
    bonus_score = 0
    resume_lower = resume_text.lower()
    
    # Bonus for having multiple sections
    sections = ['experience', 'skills', 'education', 'projects', 'certifications']
    section_bonus = sum(1 for section in sections if section in resume_lower) * 2
    
    # Bonus for technical depth
    technical_indicators = ['years', 'experience', 'proficient', 'expert', 'advanced', 'senior']
    technical_bonus = sum(1 for indicator in technical_indicators if indicator in resume_lower) * 1
    
    # Bonus for specific technologies mentioned
    tech_mentions = len([term for term in ['python', 'django', 'react', 'sql', 'git', 'docker'] if term in resume_lower])
    tech_bonus = tech_mentions * 1.5
    
    bonus_score = min(section_bonus + technical_bonus + tech_bonus, 15)  # Cap at 15 points
    
    # Weighted combination: 30% TF-IDF, 40% Semantic, 30% Keywords + Bonus
    base_score = (0.3 * tfidf_sim + 0.4 * semantic_sim + 0.3 * keyword_sim) * 100
    final_score = min(base_score + bonus_score, 100)  # Cap at 100%
    
    return {
        'final_score': round(final_score, 2),
        'tfidf_similarity': round(tfidf_sim, 4),
        'semantic_similarity': round(semantic_sim, 4),
        'keyword_similarity': round(keyword_sim, 4),
        'bonus_score': round(bonus_score, 2),
        'base_score': round(base_score, 2)
    }

# FAISS helper for storing candidate chunks (for RAG)
def create_faiss_index(embeds, dim):
    index = faiss.IndexFlatIP(dim)  # use inner product and normalized vectors
    return index

def add_to_index(index, embeds):
    # expects already normalized vectors if using IP. We'll normalize.
    import numpy as np
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms==0] = 1
    normalized = embeds / norms
    index.add(normalized)
    return index

"""
Query classification module.
Classifies user queries as simple or complex using keyword heuristics.
"""
from typing import Literal
from src.config import COMPLEX_QUERY_KEYWORDS


def classify_query(question: str) -> Literal["simple", "complex"]:
    """
    Classify a user query as simple or complex.
    
    Simple queries: Direct retrieval questions (e.g., "What errors occurred?")
    Complex queries: Analytical questions requiring aggregation or temporal analysis
                     (e.g., "How many errors over time?")
    
    Args:
        question: User's question string
        
    Returns:
        "simple" or "complex"
    """
    question_lower = question.lower()
    
    if len(question_lower.split()) > 12:
        return "complex"

    # Check if any complex query keyword is present
    for keyword in COMPLEX_QUERY_KEYWORDS:
        if keyword in question_lower:
            return "complex"
    
    return "simple"

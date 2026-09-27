"""
=============================================================================
Project Title : Development of a College FAQ Chatbot Using Natural Language Processing
Category      : College Examinations
Chatbot Name  : CampusExam AI
Backend       : Python Flask + Scikit-Learn TF-IDF & Cosine Similarity
=============================================================================
"""

import json
import os
import re
import math
from collections import Counter
from flask import Flask, request, jsonify, render_template

# Initialize Flask Application
app = Flask(__name__, template_folder="templates", static_folder="static")

# Constants & Configuration
FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "faq.json")
SIMILARITY_THRESHOLD = 0.25  # Confidence threshold for intent matching

# Common abbreviations and domain-specific normalizations
SYNONYM_MAP = {
    r"\bsem\b": "semester",
    r"\bsems\b": "semesters",
    r"\bhallticket\b": "hall ticket",
    r"\badmit card\b": "hall ticket",
    r"\badmitcard\b": "hall ticket",
    r"\bmarksheet\b": "mark sheet",
    r"\bmarks card\b": "mark sheet",
    r"\bgrade card\b": "mark sheet",
    r"\breval\b": "revaluation",
    r"\brevaln\b": "revaluation",
    r"\bre checking\b": "revaluation",
    r"\brecheck\b": "revaluation",
    r"\bre totalling\b": "retotalling",
    r"\bsupply\b": "supplementary",
    r"\bsupple\b": "supplementary",
    r"\bbacklog\b": "arrears",
    r"\bbacklogs\b": "arrears",
    r"\barrear\b": "arrears",
    r"\battandance\b": "attendance",
    r"\battendence\b": "attendance",
    r"\bcalc\b": "calculator",
    r"\bcalci\b": "calculator",
    r"\bcoe\b": "examination cell",
    r"\bexam cell\b": "examination cell",
    r"\btime table\b": "timetable",
    r"\bdate sheet\b": "timetable",
    r"\bdatesheet\b": "timetable",
    r"\bmidterm\b": "internal exam",
    r"\bmid term\b": "internal exam",
    r"\bcia\b": "internal marks",
    r"\bviva\b": "project viva",
    r"\bviva-voce\b": "project viva",
    r"\bcheating\b": "malpractice",
    r"\bcopying\b": "malpractice",
    r"\bphone\b": "mobile phone",
    r"\bmobiles\b": "mobile phone",
    r"\bcellphone\b": "mobile phone",
    r"\bsmartphone\b": "mobile phone",
    r"\bsmartphones\b": "mobile phone",
    r"\bsmartwatch\b": "smartwatch",
    r"\bsmartwatches\b": "smartwatch",
    r"\bsmart watch\b": "smartwatch",
    r"\bsmart watches\b": "smartwatch",
    r"\bpass mark\b": "passing criteria",
    r"\bpass marks\b": "passing criteria",
}

# English stop words list (crafted for examination Q&A to preserve key question cues)
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "today", "tomorrow", "yesterday", "tell", "please", "kindly", "give", "show", "know"
}

class NLPPreprocessor:
    """
    Handles text normalization, tokenization, synonym mapping, and token filtering.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Apply domain-specific synonym and abbreviation replacements
        for pattern, replacement in SYNONYM_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Remove non-alphanumeric characters (keep basic spaces and hyphens)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(text: str) -> list[str]:
        cleaned = NLPPreprocessor.clean_text(text)
        if not cleaned:
            return []
        tokens = cleaned.split()
        return tokens

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        filtered = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
        # If all tokens were filtered out (e.g. query was very short), retain original tokens
        return filtered if filtered else tokens

    @staticmethod
    def generate_ngrams(tokens: list[str], n_min: int = 1, n_max: int = 2) -> list[str]:
        """Generates unigrams and bigrams for rich context capture."""
        ngrams = []
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams


class TFIDFEngine:
    """
    Explainable TF-IDF Vectorizer and Cosine Similarity Engine.
    Uses Scikit-Learn if available, with built-in pure Python TF-IDF fallback for transparent viva explanation.
    """
    def __init__(self):
        self.vocabulary = {}
        self.idf_weights = {}
        self.doc_vectors = []
        self.corpus_data = []  # List of dicts {intent, pattern, category, answer, quick_chips}
        self.is_trained = False
        self.sklearn_vectorizer = None
        self.sklearn_doc_matrix = None

        # Try to import sklearn for standard production execution
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            sanitized_stops = list({NLPPreprocessor.clean_text(w) for w in STOP_WORDS if NLPPreprocessor.clean_text(w)})
            self.sklearn_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=sanitized_stops,
                sublinear_tf=True
            )
            self._has_sklearn = True
        except ImportError:
            self._has_sklearn = False

    def train(self, intents_data: list[dict]):
        """
        Builds the corpus and computes TF-IDF representations for all training patterns.
        """
        self.corpus_data = []
        raw_corpus_texts = []

        for intent_item in intents_data:
            intent = intent_item["intent"]
            category = intent_item.get("category", "General")
            answer = intent_item["answer"]
            quick_chips = intent_item.get("quick_chips", [])
            patterns = intent_item.get("patterns", [])

            for pattern in patterns:
                cleaned_pattern = NLPPreprocessor.clean_text(pattern)
                tokens = NLPPreprocessor.tokenize(cleaned_pattern)
                filtered_tokens = NLPPreprocessor.remove_stopwords(tokens)

                self.corpus_data.append({
                    "intent": intent,
                    "category": category,
                    "original_pattern": pattern,
                    "cleaned_pattern": cleaned_pattern,
                    "tokens": filtered_tokens,
                    "answer": answer,
                    "quick_chips": quick_chips
                })
                raw_corpus_texts.append(cleaned_pattern)

        # 1. Train Scikit-Learn Vectorizer if present
        if self._has_sklearn:
            self.sklearn_doc_matrix = self.sklearn_vectorizer.fit_transform(raw_corpus_texts)

        # 2. Build Explainable Pure-Python TF-IDF Engine
        total_docs = len(self.corpus_data)
        doc_term_freqs = []
        all_terms = set()

        for item in self.corpus_data:
            ngrams = NLPPreprocessor.generate_ngrams(item["tokens"], 1, 2)
            term_counts = Counter(ngrams)
            doc_term_freqs.append(term_counts)
            all_terms.update(term_counts.keys())

        # Create Vocabulary Index
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Calculate IDF for each term: log((1 + N) / (1 + DF)) + 1 (Smooth IDF)
        self.idf_weights = {}
        for term in self.vocabulary:
            df = sum(1 for tf_dict in doc_term_freqs if term in tf_dict)
            self.idf_weights[term] = math.log((1 + total_docs) / (1 + df)) + 1.0

        # Calculate TF-IDF vectors for all training documents
        self.doc_vectors = []
        for tf_dict in doc_term_freqs:
            vec = self._compute_vector_from_tf(tf_dict)
            self.doc_vectors.append(vec)

        self.is_trained = True

    def _compute_vector_from_tf(self, tf_dict: dict) -> dict[int, float]:
        """Calculates L2-normalized TF-IDF vector."""
        total_terms = sum(tf_dict.values())
        vector = {}
        sum_sq = 0.0

        for term, count in tf_dict.items():
            if term in self.vocabulary:
                term_idx = self.vocabulary[term]
                # Sublinear TF scaling: 1 + log(tf) if tf > 0
                tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                idf_val = self.idf_weights.get(term, 1.0)
                tfidf_val = tf_val * idf_val
                vector[term_idx] = tfidf_val
                sum_sq += tfidf_val ** 2

        # L2-Normalization
        norm = math.sqrt(sum_sq)
        if norm > 0:
            for term_idx in vector:
                vector[term_idx] /= norm

        return vector

    def _vectorize_query(self, query_text: str) -> tuple[dict[int, float], list[str]]:
        """Vectorizes a new user query."""
        cleaned = NLPPreprocessor.clean_text(query_text)
        tokens = NLPPreprocessor.tokenize(cleaned)
        filtered = NLPPreprocessor.remove_stopwords(tokens)
        ngrams = NLPPreprocessor.generate_ngrams(filtered, 1, 2)
        tf_dict = Counter(ngrams)

        vector = self._compute_vector_from_tf(tf_dict)
        return vector, filtered

    @staticmethod
    def _cosine_similarity(vec1: dict[int, float], vec2: dict[int, float]) -> float:
        """Computes dot product of two L2-normalized sparse vectors."""
        # Since vectors are L2-normalized, cosine similarity is simply the dot product
        common_indices = set(vec1.keys()) & set(vec2.keys())
        return sum(vec1[idx] * vec2[idx] for idx in common_indices)

    def predict_intent(self, user_query: str) -> dict:
        """
        Matches a user query against the FAQ knowledge base using TF-IDF + Cosine Similarity.
        """
        if not self.is_trained:
            return {
                "error": "NLP Engine is not trained yet.",
                "status": "error"
            }

        cleaned_query = NLPPreprocessor.clean_text(user_query)
        if not cleaned_query:
            return {
                "intent": "empty_query",
                "answer": "Please enter a valid question about college examinations.",
                "confidence": 0.0,
                "confidence_percentage": "0%",
                "status": "empty"
            }

        query_tokens = NLPPreprocessor.tokenize(cleaned_query)
        filtered_tokens = NLPPreprocessor.remove_stopwords(query_tokens)

        # Primary or Hybrid Matching
        best_match_idx = -1
        highest_similarity = 0.0

        if self._has_sklearn and self.sklearn_vectorizer and self.sklearn_doc_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.sklearn_vectorizer.transform([cleaned_query])
            similarities = cosine_similarity(query_vec, self.sklearn_doc_matrix).flatten()
            best_match_idx = int(similarities.argmax())
            highest_similarity = float(similarities[best_match_idx])
        else:
            # Pure Python Cosine Similarity computation
            query_vec, _ = self._vectorize_query(user_query)
            for idx, doc_vec in enumerate(self.doc_vectors):
                sim = self._cosine_similarity(query_vec, doc_vec)
                if sim > highest_similarity:
                    highest_similarity = sim
                    best_match_idx = idx

        # Determine matched terms for transparency
        matched_terms = [t for t in filtered_tokens if t in self.vocabulary]

        # Check against similarity threshold
        if highest_similarity >= SIMILARITY_THRESHOLD and best_match_idx >= 0:
            matched_doc = self.corpus_data[best_match_idx]
            return {
                "status": "matched",
                "intent": matched_doc["intent"],
                "category": matched_doc["category"],
                "answer": matched_doc["answer"],
                "confidence": round(highest_similarity, 4),
                "confidence_percentage": f"{min(int(highest_similarity * 100), 100)}%",
                "matched_pattern": matched_doc["original_pattern"],
                "quick_chips": matched_doc["quick_chips"],
                "nlp_breakdown": {
                    "original_query": user_query,
                    "cleaned_query": cleaned_query,
                    "raw_tokens": query_tokens,
                    "processed_tokens": filtered_tokens,
                    "matched_vocabulary_terms": matched_terms,
                    "similarity_score": round(highest_similarity, 4),
                    "intent": matched_doc["intent"],
                    "category": matched_doc["category"]
                }
            }
        else:
            # Fallback response with helpful topic guidance
            fallback_chips = ["Exam Dates", "Hall Ticket", "Results", "Revaluation", "Attendance", "Exam Rules"]
            return {
                "status": "fallback",
                "intent": "fallback_unknown",
                "category": "General",
                "answer": "I'm not completely sure about that question. Please try asking about examination dates, hall tickets, results, attendance, revaluation, or exam registration.",
                "confidence": round(highest_similarity, 4),
                "confidence_percentage": f"{int(highest_similarity * 100)}%",
                "matched_pattern": None,
                "quick_chips": fallback_chips,
                "nlp_breakdown": {
                    "original_query": user_query,
                    "cleaned_query": cleaned_query,
                    "raw_tokens": query_tokens,
                    "processed_tokens": filtered_tokens,
                    "matched_vocabulary_terms": matched_terms,
                    "similarity_score": round(highest_similarity, 4),
                    "intent": "fallback_unknown",
                    "category": "Unknown"
                }
            }


# Global Instance of NLP Engine
nlp_engine = TFIDFEngine()

def initialize_faq_model():
    """Loads FAQ data and trains the NLP engine."""
    if not os.path.exists(FAQ_DATA_PATH):
        raise FileNotFoundError(f"FAQ dataset file not found at {FAQ_DATA_PATH}")

    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    intents = faq_data.get("intents", [])
    nlp_engine.train(intents)
    print(f"[CampusExam AI] NLP Engine initialized with {len(intents)} intents and {len(nlp_engine.corpus_data)} training patterns.")

# Initialize model upon loading
try:
    initialize_faq_model()
except Exception as e:
    print(f"[Warning] Failed to initialize model at startup: {e}")


# =============================================================================
# FLASK WEB ROUTES & REST API
# =============================================================================

@app.route("/")
def index():
    """Renders the main student chatbot interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main Chat API Endpoint.
    Expects JSON: { "message": "When are semester exams?" }
    Returns JSON with intent, answer, confidence score, and NLP breakdown.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "message" not in data:
            return jsonify({
                "status": "error",
                "answer": "Invalid request. Please provide a 'message' field in JSON.",
                "confidence": 0.0,
                "confidence_percentage": "0%",
                "intent": "error"
            }), 400

        user_message = str(data.get("message", "")).strip()

        if not user_message:
            return jsonify({
                "status": "empty",
                "answer": "Please type an examination-related question.",
                "confidence": 0.0,
                "confidence_percentage": "0%",
                "intent": "empty_query"
            }), 200

        # Perform NLP prediction
        result = nlp_engine.predict_intent(user_message)
        return jsonify(result), 200

    except Exception as ex:
        return jsonify({
            "status": "error",
            "answer": f"An error occurred while processing your question: {str(ex)}",
            "confidence": 0.0,
            "confidence_percentage": "0%",
            "intent": "server_error"
        }), 500


@app.route("/api/faq", methods=["GET"])
def get_faq_directory():
    """Returns the full list of FAQ intents and sample questions for browsing."""
    try:
        with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        return jsonify(faq_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nlp-info", methods=["GET"])
def get_nlp_info():
    """Returns metadata about the NLP model for viva inspection."""
    return jsonify({
        "project_title": "Development of a College FAQ Chatbot Using Natural Language Processing",
        "category": "College Examinations",
        "chatbot_name": "CampusExam AI",
        "nlp_technique": "TF-IDF (Term Frequency - Inverse Document Frequency) + Cosine Similarity",
        "ngram_range": "(1, 2) - Unigrams and Bigrams",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "total_intents": len(set(d["intent"] for d in nlp_engine.corpus_data)),
        "total_training_patterns": len(nlp_engine.corpus_data),
        "vocabulary_size": len(nlp_engine.vocabulary),
        "features": [
            "Text Lowercasing & Punctuation Stripping",
            "Synonym & Abbreviation Normalization (sem, hallticket, reval, etc.)",
            "Stop Words Removal",
            "Unigram + Bigram Extraction",
            "L2-Normalized TF-IDF Vector Space Modeling",
            "Cosine Similarity Distance Metric",
            "Confidence Scoring & Fallback Thresholding"
        ]
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "app": "CampusExam AI",
        "trained": nlp_engine.is_trained
    }), 200


# Application Entry Point
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" CampusExam AI - College FAQ Chatbot")
    print(" Running locally on http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)

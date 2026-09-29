"""
Marathi NLP Preprocessing Module
Provides reusable text cleaning, normalization, stopword removal, and tokenization for Marathi text.
Used identically during training and inference to prevent domain/pipeline mismatch.
"""

import re
import unicodedata
from typing import List, Tuple

# Comprehensive Marathi stopwords list (grammatical particles, pronouns, auxiliary verbs)
MARATHI_STOPWORDS = {
    "आणि", "व", "किंवा", "पण", "परंतु", "नाही", "नाहीत", "आहे", "आहेत",
    "होते", "होती", "होता", "होऊ", "झाले", "झाली", "झाला", "झालेला",
    "झालेली", "झालेले", "करणे", "केले", "केली", "केला", "या", "ती",
    "तो", "ते", "त्या", "त्यांच्या", "त्यांना", "त्यांनी", "त्यामुळे",
    "त्यात", "यांना", "यांनी", "यांच्या", "यात", "यामुळे", "याबाबत",
    "हा", "ही", "हे", "ह्या", "इथे", "तिथे", "येथे", "तेथे", "कुठे",
    "कधी", "कसे", "कशी", "कसा", "कोण", "काय", "का", "सर्व", "काही",
    "इतर", "अनेक", "असा", "अशी", "असे", "अशा", "तसा", "तशी", "तसे",
    "तशा", "तसेच", "सुद्धा", "देखील", "मात्र", "तरी", "तरीही", "म्हणून",
    "कारण", "जर", "तर", "मग", "आता", "नंतर", "पूर्वी", "पुढे", "मागे",
    "वर", "खाली", "मध्ये", "बाहेर", "जवळ", "दूर", "साठी", "पासून",
    "कडून", "बद्दल", "विषयी", "ने", "नी", "मुळे", "द्वारे", "इत्यादी",
    "इ", "असेल", "असतील", "केल्या", "केले", "दिला", "दिली", "दिले",
    "घेतला", "घेतली", "घेतले", "सांगितले", "म्हणाले", "केलाय", "करतेय", "करतोय"
}

# Devanagari Unicode Range: \u0900-\u097F
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

def normalize_unicode(text: str) -> str:
    """Normalizes Unicode characters to standard NFKC representation."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)

def remove_urls_and_markup(text: str) -> str:
    """Removes web URLs, HTML tags, and mentions."""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    return text

def remove_special_characters(text: str) -> str:
    """
    Cleans punctuation and special symbols while strictly preserving
    Devanagari characters, English alphanumerics (for abbreviations like AI, 5G, IPL),
    and essential spaces.
    """
    # Replace traditional Devanagari danda and double danda with spaces
    text = text.replace("।", " ").replace("॥", " ")
    
    # Remove standard punctuation and symbols
    # Keep Devanagari (\u0900-\u097F), Latin letters (a-zA-Z), and digits (0-9, \u0966-\u096F)
    text = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', ' ', text)
    
    return text

def tokenize_marathi(text: str) -> List[str]:
    """Tokenizes cleaned Marathi text into words by whitespace."""
    return [token for token in text.split() if token]

def clean_marathi_text(text: str, remove_stops: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline for Marathi news text.
    Preserves Devanagari script integrity.
    
    Parameters:
        text (str): Raw input headline or news text.
        remove_stops (bool): Whether to filter out Marathi stopwords.
        
    Returns:
        str: Preprocessed, normalized, and cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode normalization
    text = normalize_unicode(text)
    
    # 2. Lowercase any English characters in abbreviations
    text = text.lower()
    
    # 3. Strip URLs, markup, email/usernames
    text = remove_urls_and_markup(text)
    
    # 4. Strip punctuation and non-Devanagari / non-alphanumeric noise
    text = remove_special_characters(text)
    
    # 5. Tokenization
    tokens = tokenize_marathi(text)
    
    # 6. Stopwords removal (optional, defaults to True)
    if remove_stops:
        tokens = [t for t in tokens if t not in MARATHI_STOPWORDS and len(t) > 1]
    
    # 7. Reassemble tokens with clean single spaces
    return " ".join(tokens).strip()

def validate_marathi_input(text: str) -> Tuple[bool, str]:
    """
    Validates user input headline for correctness.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message_or_hint)
    """
    if not text or not text.strip():
        return False, "Please enter a Marathi news headline. (कृपया मराठी बातमीचे शीर्षक प्रविष्ट करा.)"
    
    stripped = text.strip()
    if len(stripped) < 5:
        return False, "Headline is too short. Please provide a complete Marathi headline. (बातमीचे शीर्षक खूप लहान आहे.)"
    
    has_devanagari = bool(DEVANAGARI_PATTERN.search(stripped))
    if not has_devanagari:
        return False, "Please enter the headline in Marathi / Devanagari script. (कृपया देवनागरी/मराठी लिपीत बातमी प्रविष्ट करा.)"
        
    return True, ""

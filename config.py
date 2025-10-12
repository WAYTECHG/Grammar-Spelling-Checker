# config.py
CONFIG = {
    # Layer 1 settings
    "spelling": {
        "max_edit_distance": 2,
        "suggestion_threshold": 0.3,
        "max_suggestions": 5,
        "use_phonetic_matching": True
    },
    "grammar": {
        "ngram_size": 3,
        "rule_confidence_threshold": 0.6,
        "use_dependency_parsing": True,
        "corenlp_url": "http://127.0.0.1:9002"
    },
    
    # Layer 2 settings
    "ai_correction": {
        "enabled": True,
        "models": ["bart", "seq2seq"],  # Available models
        "confidence_threshold": 0.7,
        "max_context_length": 2000,
        "temperature": 0.3
    },
    
    # System settings
    "server": {
        "port": 5000,
        "debug": False,
        "max_upload_size_mb": 10,
        "allowed_file_types": [".txt", ".docx", ".pdf"]
    }
} 
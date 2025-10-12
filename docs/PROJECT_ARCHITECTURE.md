# Enhanced Spelling and Grammar Checker - Project Architecture

## 1. System Overview

The Enhanced Spelling and Grammar Checker is a two-layer text correction system that combines classical NLP techniques with modern AI approaches to provide comprehensive text correction capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│                 (HTML/JS via Flask)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend Server                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Layer 1: Classical Engine │   │ Layer 2: AI-Powered       │
│ - Spelling Checker        │──▶│   Correction              │
│ - Grammar Checker         │   │ - T5 Transformer          │
│ - CoreNLP Integration     │   │ - Seq2Seq RNN Model       │
└───────────────────────────┘   └───────────────────────────┘
                                            │
                                            ▼
                            ┌─────────────────────────────────┐
                            │        Post-Processing          │
                            │   (Merge & Rank Suggestions)    │
                            └─────────────────────────────────┘
                                            │
                                            ▼
                            ┌─────────────────────────────────┐
                            │        Result Display           │
                            └─────────────────────────────────┘
```

## 2. Component Details

### 2.1 User Interface (Frontend)

**Implementation:** HTML/CSS/JavaScript served via Flask templates

**Features:**

- Text input box for direct entry
- File upload capability (supporting .txt, .docx, .pdf)
- Mode selection:
  - Quick Check (Layer 1 only)
  - Deep Check (Layer 1 + Layer 2 AI)
- Visual highlighting of errors and suggestions
- Confidence score display for each suggestion
- Option to accept/reject individual suggestions
- Export corrected text functionality

### 2.2 Flask Backend Server

**Routes:**

- `/` (GET): Serve the main interface
- `/check` (POST): Process text with Layer 1 checking (Prints: "Running Quick Check - Layer 1")
- `/deepcheck` (POST): Process text with both Layer 1 and Layer 2 (Prints: "Running Deep Check - Layer 1 + Layer 2")
- `/bart` (POST): Endpoint for BART model inference (Prints: "Using BART Transformer for correction")
- `/seq2seq` (POST): Endpoint for Seq2Seq RNN model inference (Prints: "Using Seq2Seq RNN for correction")
- `/status` (GET): Return system status (CoreNLP server, AI models)
- `/accept` (POST): Apply selected corrections
- `/download` (GET): Download corrected text

**Implementation:**

```python
@app.route('/check', methods=['POST'])
def check_text():
    print("Running Quick Check - Layer 1")
    text = request.form.get('text', '')
    # Process with Layer 1
    spelling_checker = EnhancedSpellingChecker()
    grammar_checker = EnhancedGrammarChecker()
  
    # Get corrections
    spelling_corrected = spelling_checker.correct_spelling(text)
    grammar_issues = grammar_checker.check_grammar(spelling_corrected)
  
    return jsonify({
        'corrected_text': spelling_corrected,
        'grammar_issues': grammar_issues,
        'confidence_scores': calculate_confidence(grammar_issues)
    })
```

### 2.3 Layer 1: Classical Grammar & Spelling Checking Engine

#### 2.3.1 Spelling Checker Submodule

**Datasets:**

- Birkbeck corpus (missp.dat, aspell.dat, wikipedia.dat)
- SCOWL word list (scowl.txt)
- Google 10,000 English words (google-10000-english-no-swears.txt)

**Methods:**

- Edit Distance (Levenshtein algorithm) (Prints: "Calculating edit distance" + "Current text: [text after Edit Distance correction]")
- Phonetic matching:
  - Soundex (Prints: "Applying Soundex phonetic matching" + "Current text: [text after Soundex correction]")
  - Metaphone (Prints: "Applying Metaphone phonetic matching" + "Current text: [text after Metaphone correction]")
- SymSpell algorithm for faster candidate generation
- Dictionary checking (Prints: "Checking in dictionaries" + "Current text: [text after dictionary check]")
- Candidate scoring system:
  - Edit distance weight: 40%
  - Length similarity weight: 20%
  - Phonetic similarity weight: 20%
  - Prefix matching weight: 20%
- Common contractions handling

**Key Functions:**

- `load_dictionaries()`: Load and index all spelling datasets
- `get_candidates(word)`: Generate correction candidates
- `correct_spelling(text)`: Process full text for spelling correction

#### 2.3.2 Grammar Checker Submodule

**Grammar Corpora:**

- JFLEG dataset
- CoNLL-14 dataset
- FCE dataset
- Lang-8 dataset

**Methods:**

- Rule-based pattern matching (regex) (Prints: "Applying rule-based grammar check with Regex" + "Current text: [text after Regex correction]")
- N-gram language model with Laplace smoothing (Prints: "Calculating sentence probability using N-gram model" + "Current text: [text after N-gram correction]")
- POS tagging for grammatical analysis
- Stanford CoreNLP integration for:
  - Dependency parsing (Prints: "Performing dependency parsing with Stanford CoreNLP" + "Current text: [text after Dependency Parsing correction]")
  - Named entity recognition
  - Sentence structure analysis

**Grammar Rules Categories:**

- Subject-verb agreement
- Article usage
- Preposition selection
- Tense consistency
- Common homophones (their/there/they're)
- Countable/uncountable nouns
- Gerund vs. infinitive usage
- Punctuation rules

**Key Functions:**

- `load_grammar_datasets()`: Load and prepare grammar datasets
- `train_ngram_model()`: Build statistical language model
- `check_grammar(text)`: Identify grammar issues with confidence scores
- `get_final_correction(text)`: Apply combined corrections

**Decision Logic:**

- If confidence < 0.6 OR
- If complex sentence structure detected OR
- If user selected "Deep Check"
  → Escalate to Layer 2 (Prints: "Layer 1 output insufficient, triggering Layer 2")
- Otherwise (Prints: "Layer 1 output sufficient" + "Final text after Layer 1: [final Layer 1 text]")

### 2.4 Layer 2: AI-Powered Correction

#### 2.4.1 BART Transformer Model

**Model Options:**

- BART fine-tuned for grammar correction (HuggingFace Model)

**Implementation:**

```python
def apply_bart_correction(text):
    print("Running BART correction")
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
  
    # Generate correction
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512
    )
  
    # Decode output
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Text after BART correction: {corrected_text}")
    return corrected_text
```

#### 2.4.2 Seq2Seq RNN Model

**Model Implementation:**

- Custom trained Seq2Seq RNN for grammar correction

**Implementation:**

```python
def apply_seq2seq_correction(text):
    print("Running Seq2Seq RNN correction")
    # Process with Seq2Seq model
    # ...
  
    print(f"Text after Seq2Seq RNN correction: {corrected_text}")
    return corrected_text
```

### 2.5 Post-Processing

**Functions:**

- Merge Layer 1 and Layer 2 suggestions
- Remove duplicate or conflicting corrections
- Rank suggestions by confidence score
- Filter suggestions below threshold
- Maintain correction history

**Implementation:**

```python
def merge_corrections(layer1_corrections, layer2_corrections):
    print("Aggregating Layer 2 corrections")
    # Prioritize high-confidence corrections from either layer
    # Resolve conflicts by preferring Layer 2 for complex issues
    # Return unified correction list
    print(f"Final aggregated text: {aggregated_text}")
    pass
```

### 2.6 Result Display

**Features:**

- Side-by-side comparison (original vs. corrected)
- Highlighted corrections with color coding:
  - Red: Spelling errors
  - Blue: Grammar errors
  - Yellow: Style suggestions
- Hover tooltips showing:
  - Error description
  - Confidence score
  - Correction source (Layer 1/2)
- Accept/reject buttons for each suggestion
- Statistics summary (error counts by type)
- Download options (plain text, HTML, PDF)

## 3. Integration Points

### 3.1 Stanford CoreNLP Server

**Setup:**

- Java 8+ required
- Server runs on port 9002
- Annotators needed: tokenize, ssplit, pos, lemma, ner, parse, depparse
- Memory allocation: 4GB minimum

**Integration Code:**

```python
from nltk.parse.corenlp import CoreNLPParser

# Initialize parser
parser = CoreNLPParser(url='http://127.0.0.1:9002')

# Parse sentence
parse_tree = next(parser.raw_parse(sentence))
```

**Startup Script:**

```bash
#!/bin/bash
cd stanford-corenlp-4.5.10
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 15000
```

### 3.2 Processing Flow

```
User Input (Text/File)
       │
       ▼
Pre-processing (Tokenization, Extraction)
  (Print: "Pre-processing input")
       │
       ▼
Layer 1: Classical Correction Engine
       │
       ├─► Spelling Correction
       │      ├─ Dictionaries (Birkbeck, SCOWL) (Print: "Checking in dictionaries") (Print: "Current text: [text after dictionary check]")
       │      ├─ Soundex Matching (Print: "Applying Soundex phonetic matching") (Print: "Current text: [text after Soundex correction]")
       │      ├─ Metaphone Matching (Print: "Applying Metaphone phonetic matching") (Print: "Current text: [text after Metaphone correction]")
       │      └─ Edit Distance Calculation (Print: "Calculating edit distance") (Print: "Current text: [text after Edit Distance correction]")
       │
       └─► Grammar Correction
              ├─ Rule-based Regex (Print: "Applying rule-based grammar check with Regex") (Print: "Current text: [text after Regex correction]")
              ├─ N-gram Model Probability Calculation (Print: "Calculating sentence probability using N-gram model") (Print: "Current text: [text after N-gram correction]")
              └─ Stanford CoreNLP Dependency Parsing (Print: "Performing dependency parsing with Stanford CoreNLP") (Print: "Current text: [text after Dependency Parsing correction]")
       │
       ▼
Layer 2 Trigger? (Confidence < threshold / Deep Mode)
       │
       ├─► NO: Display Layer 1 Output (Print: "Layer 1 output sufficient") (Print: "Final text after Layer 1: [final Layer 1 text]")
       │
       └─► YES:
               │
               ▼
         Layer 2: AI Correction Engine
               ├─► BART Transformer (HuggingFace Model) (Print: "Running BART correction") (Print: "Text after BART correction: [text]")
               └─► Seq2Seq RNN (Custom Trained Model) (Print: "Running Seq2Seq RNN correction") (Print: "Text after Seq2Seq RNN correction: [text]")
               │
               ▼
         Correction Aggregation (Print: "Aggregating Layer 2 corrections") (Print: "Final aggregated text: [aggregated text]")
               │
               ▼
         Result Display (Frontend)
```

## 4. Performance Considerations

### 4.1 Optimization Strategies

- **Caching:**
  - Cache frequent spelling corrections
  - Store n-gram probabilities
  - Memorize dependency parse results

- **Batch Processing:**
  - Process multiple sentences in parallel
  - Batch API requests to external LLMs

- **Resource Management:**
  - Lazy loading of large datasets
  - Release memory for unused models
  - Implement timeout handling for CoreNLP requests

### 4.2 Scalability

- **Horizontal Scaling:**
  - Stateless design for Flask backend
  - Load balancing for multiple instances
  - Distributed processing for large documents

- **Vertical Scaling:**
  - Memory optimization for dataset loading
  - GPU acceleration for transformer models
  - Efficient indexing of spelling dictionaries

## 5. Configuration Options

```python
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
```

## 6. Testing Strategy

### 6.1 Unit Tests

- Test individual components:
  - Levenshtein distance calculation
  - Phonetic matching algorithms
  - N-gram model probability calculation
  - Grammar rule matching

### 6.2 Integration Tests

- Test component interactions:
  - Spelling + Grammar pipeline
  - CoreNLP integration
  - Layer 1 + Layer 2 merging

### 6.3 End-to-End Tests

- Test complete workflow:
  - File upload → processing → results display
  - API request handling
  - Error handling and recovery

### 6.4 Benchmark Tests

- Performance metrics:
  - Processing time per 1000 words
  - Memory usage
  - Correction accuracy on standard datasets
  - API response times

## 7. Deployment Instructions

### 7.1 Prerequisites

- Python 3.8+
- Java 8+ (for CoreNLP)
- 8GB RAM minimum
- CUDA-compatible GPU (optional, for transformer models)

### 7.2 Installation Steps

1. Clone repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Download CoreNLP: `./download_corenlp.sh`
4. Download datasets: `python download_datasets.py`
5. Set environment variables: `cp .env.example .env` and edit
6. Start CoreNLP server: `./start_corenlp.sh`
7. Start application: `python app.py`

### 7.3 Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install Java for CoreNLP
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Copy application
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download resources
RUN python download_datasets.py

# Expose ports
EXPOSE 5000 9002

# Start services
CMD ["./start_app.sh"]
```

## 8. Development Roadmap

### Phase 1: Core Implementation
- ✅ Implement basic spelling checker
- ✅ Implement rule-based grammar checker
- ✅ Integrate CoreNLP for dependency parsing
- ✅ Create Flask web interface

### Phase 2: Enhanced Features
- ⬜ Add context-aware spelling correction
- ⬜ Expand grammar rule set
- ⬜ Improve confidence scoring system

### Phase 3: AI Integration
- ⬜ Integrate BART transformer-based correction model
- ⬜ Integrate Seq2Seq RNN model
- ⬜ Implement post-processing and suggestion merging
- ⬜ Enhance user interface with interactive corrections

### Phase 4: Optimization & Deployment
- ⬜ Optimize performance and resource usage
- ⬜ Add comprehensive testing suite
- ⬜ Create deployment documentation
- ⬜ Implement monitoring and logging

## 9. Comprehensive Project Task Management Plan

#### 9.1 Project Planning

* 9.1.1 Requirements Gathering
* 9.1.2 Dataset Collection (Spelling & Grammar)
* 9.1.3 Model Selection Confirmation (BART, Seq2Seq RNN)

#### 9.2 Backend Development

* 9.2.1 Pre-processing

  * 9.2.1.1 Tokenization function
  * 9.2.1.2 File/Text extraction function
* 9.2.2 Layer 1 Classical Correction Engine

  * 9.2.2.1 Spelling Checker

    * 9.2.2.1.1 Dictionary Checking
    * 9.2.2.1.2 Soundex Matching
    * 9.2.2.1.3 Metaphone Matching
    * 9.2.2.1.4 Edit Distance Calculation
  * 9.2.2.2 Grammar Checker

    * 9.2.2.2.1 Regex Grammar Rules
    * 9.2.2.2.2 N-gram Model Training & Inference
    * 9.2.2.2.3 Stanford CoreNLP Parsing
* 9.2.3 Layer 2 AI Correction Engine

  * 9.2.3.1 BART Transformer Integration
  * 9.2.3.2 Seq2Seq RNN Model Integration
  * 9.2.3.3 Correction Aggregation Logic
* 9.2.4 Debugging & Logging

  * 9.2.4.1 Print statement verification after each stage

#### 9.3 Frontend Development

* 9.3.1 UI Wireframe & Design

  * 9.3.1.1 Text Input Box
  * 9.3.1.2 File Upload Button
  * 9.3.1.3 Mode Selection (Radio Buttons)
  * 9.3.1.4 Result Display Area
* 9.3.2 Implementation

  * 9.3.2.1 Text Box Development
  * 9.3.2.2 File Upload Feature
  * 9.3.2.3 Mode Selection Mechanism
  * 9.3.2.4 Displaying Results

#### 9.4 API & Backend Integration

* 9.4.1 Flask Route Definitions (`/check`, `/deepcheck`, `/bart`, `/seq2seq`)
* 9.4.2 JSON Response Formatting

#### 9.5 Testing & Validation

* 9.5.1 Unit Testing Backend Components

  * 9.5.1.1 Preprocessing
  * 9.5.1.2 Spelling Techniques
  * 9.5.1.3 Grammar Techniques
  * 9.5.1.4 AI Model Inference
* 9.5.2 Unit Testing Frontend Components
* 9.5.3 Integration Testing (Backend + Frontend)
* 9.5.4 User Acceptance Testing (UAT)

#### 9.6 Documentation

* 9.6.1 Architecture Document (This File)
* 9.6.2 User Manual
* 9.6.3 Developer Manual

#### 9.7 Deployment

* 9.7.1 Local Server Deployment
* 9.7.2 Cloud Deployment (Optional)

#### 9.8 Project Management & Reporting

* 9.8.1 Weekly Progress Tracking
* 9.8.2 Milestone Review
* 9.8.3 Final Report Submission

## 10. Required Tools and Libraries

* **Programming Language**: Python 3.x
* **Web Framework**: Flask
* **Frontend**: HTML5, CSS3, JavaScript, Bootstrap, jQuery
* **Spelling Correction Libraries**: Custom Levenshtein Distance, Soundex, Metaphone implementations
* **Grammar Correction Libraries**: NLTK (for N-gram), Stanford CoreNLP
* **Deep Learning Models**: HuggingFace Transformers (BART), TensorFlow/PyTorch (Seq2Seq RNN)
* **Testing Tools**: PyTest, Flask-Testing, Selenium (optional for frontend)

## 11. Unit Test Plan

| Component                | Test Case Description                                  | Expected Output                                                                          |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Pre-processing           | Tokenization and extraction of input text              | Tokens printed, no errors                                                                |
| Spelling - Dictionary    | Word found or not in dictionary                        | Print "Checking in dictionaries" + status + current text                                 |
| Spelling - Soundex       | Correct phonetic matches generated                     | Print "Applying Soundex phonetic matching" + candidates + current text                   |
| Spelling - Metaphone     | Correct phonetic matches generated                     | Print "Applying Metaphone phonetic matching" + candidates + current text                 |
| Spelling - Edit Distance | Proper edit distance calculated for each candidate     | Print "Calculating edit distance" + scores + current text                                |
| Grammar - Regex          | Regex patterns applied correctly on sentence structure | Print "Applying rule-based grammar check with Regex" + matches + current text            |
| Grammar - N-gram         | Sentence probability calculated without errors         | Print "Calculating sentence probability using N-gram model" + probability + current text |
| Grammar - Dependency     | Dependency parsing result valid from CoreNLP server    | Print "Performing dependency parsing with Stanford CoreNLP" + parse tree + current text  |
| Layer 2 Trigger Logic    | Proper trigger condition for BART/Seq2Seq models       | Print decision path (Layer 1 sufficient/Deep check triggered)                            |
| BART Model Inference     | BART inference called with sentence                    | Print "Running BART correction" + result + current text                                  |
| Seq2Seq Model Inference  | Seq2Seq inference called with sentence                 | Print "Running Seq2Seq RNN correction" + result + current text                           |
| Post-Processing          | Suggestions aggregated correctly                       | Print "Aggregating final corrections" + final aggregated text                            |

## 12. Maintenance Guidelines

### 12.1 Dataset Updates

- Schedule quarterly updates of spelling dictionaries
- Add new grammar rules based on user feedback
- Retrain n-gram models with new corpus data

### 12.2 Model Updates

- Monitor transformer model performance
- Update BART and Seq2Seq RNN models as needed
- Fine-tune models with user correction data

### 12.3 Monitoring

- Track API usage and performance
- Monitor error rates and correction accuracy
- Implement user feedback collection
- Analyze common error patterns

---
*This architecture document serves as a blueprint for the Enhanced Spelling and Grammar Checker project. It should be updated as the project evolves and new requirements emerge.*

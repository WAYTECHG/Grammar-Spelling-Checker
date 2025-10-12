# NLP Combined Corrector

A unified text correction pipeline that combines spelling and grammar correction using classical NLP techniques.

## Overview

The NLP Combined Corrector integrates the spelling checker and grammar checker components into a single, efficient pipeline for text correction. It provides a unified interface for correcting both spelling and grammar errors in text, with support for multi-pass correction to handle complex cases.

## Features

- **Unified Correction Pipeline**: Combines spelling and grammar correction in a single process
- **Lazy Loading**: Components are loaded only when needed to improve startup time
- **Detailed Correction Information**: Option to include detailed information about corrections
- **Multi-Pass Correction**: Support for multiple correction passes to handle complex cases
- **Performance Metrics**: Tracking of processing time for each component
- **Multiple Interfaces**:
  - Python API for direct integration
  - Command-line interface for batch processing
  - Flask API for web applications

## Components

The system consists of two main correction layers:

1. **Spelling Checker** (`EnhancedSpellingChecker`):
   - Dictionary-based checking
   - Edit distance calculation
   - Phonetic matching (Soundex, Metaphone)
   - Common misspelling handling

2. **Grammar Checker** (`EnhancedGrammarChecker`):
   - Rule-based pattern matching
   - N-gram language model
   - Stanford CoreNLP integration for dependency parsing
   - Homophone detection and correction

## Installation

### Prerequisites

- Python 3.8+
- Java 8+ (for Stanford CoreNLP)
- 8GB RAM minimum

### Setup

1. **Navigate to the Project Directory**:
   Ensure you are in the root directory of the `nlp_combine` project. If you received the files as a zip, extract them and navigate into the extracted folder.

2. **Create and Activate a Python Virtual Environment**:
   It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with system-wide Python packages.

   For **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   For **Windows (Command Prompt)**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

   For **Windows (PowerShell)**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Python dependencies**:
   Install all the necessary Python libraries required for the project using `pip`. These are listed in the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```
   This command will install libraries such as Flask, NLTK, Keras, Hugging Face Transformers, and other essential NLP and deep learning packages.

4. **Download and Start Stanford CoreNLP server**:
   The `EnhancedGrammarChecker` component relies on the Stanford CoreNLP server for advanced linguistic analysis (e.g., dependency parsing, POS tagging). You need to download and start this server before running the application.

   For **macOS/Linux**:
   ```bash
   ./start_corenlp.sh
   ```
   This script will download the Stanford CoreNLP library if it's not already present and then start the server, which typically runs on `http://127.0.0.1:9002`.

   For **Windows**:
   The `start_corenlp.sh` script is designed for Unix-like systems. For Windows users, the Stanford CoreNLP library is already included in the `stanford-corenlp-4.5.10/` directory within this project. You only need to start the server.
   
   a. **Start Server**: Open your command prompt or PowerShell, navigate to the `stanford-corenlp-4.5.10/` directory, and run the server using Java:
      ```cmd
      java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 15000
      ```
      Ensure Java 8 or higher is installed and configured correctly on your system.

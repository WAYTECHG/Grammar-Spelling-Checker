# Documentation Work Breakdown

This document outlines the division of documentation tasks among the six team members, including specific file references and a detailed scope to ensure a clear, comprehensive, and logically ordered final report.

---

### Team Member 1: Project Overview, Setup, & Data Pipelines

- **Objective**: To provide a high-level entry point for new developers and to document the data preparation processes.
- **Key Files**: `README.md`, `requirements.txt`, `start_corenlp.sh`, `config.py`, `data/`, `scripts/`
- **Detailed Scope**:
  - **`README.md`**:
    - **Project Vision**: Write a concise introduction explaining what the project does.
    - **High-Level Architecture**: Create a section with a diagram showing the main components.
    - **Features**: List the core functionalities.
  - **Setup Guide**: Create a detailed setup guide explaining:
    - Virtual environment setup.
    - Dependency installation.
    - How to run the Stanford CoreNLP server.
  - **`config.py`**: Document the purpose of each configuration variable.
  - **Data Pipelines**:
    - **Data Sources**: Document all data sources located in the `data/` directory, specifying their purpose and origin:
      - **`data/grammar/`**:
        - `conll14.json`: A standard grammar correction dataset, often used for evaluating grammatical error correction (GEC) systems.
        - `fce/` (First Certificate in English corpus): Contains essays written by learners and annotated with grammatical errors. It's a key resource for the BEA2019 shared task on GEC, providing raw JSON and processed M2 formats.
        - `jfleg_test.parquet`, `jfleg_validation.parquet`: JFLEG (Joint Factual Language Evaluation of Grammaticality) datasets, used for evaluating the fluency and grammaticality of generated text, particularly for grammar correction models.
        - `lang-8-en-1.0/`: The Lang-8 corpus, a collection of English sentences written by non-native speakers with corrections by native speakers, commonly used for training and fine-tuning grammar correction models.
      - **`data/spelling/`**:
        - `aspell.dat`: An Aspell dictionary file, often used as a base for spell-checking dictionaries.
        - `google-10000-english-no-swears.txt`: A list of the 10,000 most common English words (excluding swear words), valuable for frequency-based spell-checking and dictionary creation.
        - `missp.dat`: A dataset containing common misspellings and their correct forms, used for training and testing spelling correction algorithms.
        - `scowl.txt`: The SCOWL (Words for Unix spell checkers) word list, providing a comprehensive English dictionary for spelling validation.
        - `wikipedia.dat`: A large vocabulary dataset derived from Wikipedia, useful for expanding dictionary coverage and improving correction accuracy.
    - **Processing Scripts**: Document each script in the `scripts/` directory.
  - **Testing Strategy**:
    - Add a section to the `README.md` explaining the project's testing strategy.
    - Provide instructions on how to run all tests.
    - Note which test files are considered obsolete.

---

### Team Member 2: Layer 1 - Spelling Checker

- **Objective**: To document the rule-based spelling correction module, explaining its components, logic, and how to extend it.
- **Key Files**: `layer1/spelling_checker.py`, `data/spelling/`, `scripts/prepare_spelling_dict.py`, `tests/test_layer1.py`
- **Detailed Scope**:
  - **`spelling_checker.py`**:
    - **Class Overview**: Describe the `EnhancedSpellingChecker` class, its purpose, and initialization, emphasizing its role in fast and accurate spelling correction.
    - **Dictionary Loading**: Explain how `load_dictionary()` works. Detail that it loads `frequency_dictionary_clean.txt` into SymSpell, and that this dictionary is crucial for the correction process as it contains correctly spelled words and their frequencies.
    - **Correction Algorithm**: Detail the `correct_spelling` method, explaining that it uses the SymSpell algorithm to generate spelling suggestions based on edit distance. Further, explain the custom scoring logic in `_calculate_score`, specifically how it prioritizes suggestions based on a weighted combination of edit distance and word frequency to select the most probable correction.
  - **Data Files**: Document the contents and purpose of the files in `data/spelling/`. Explain that `frequency_dictionary_clean.txt` is the primary dictionary used by SymSpell, and `missp.dat` is a source of common misspellings for training or testing.
  - **Scripts**: Explain how `scripts/prepare_spelling_dict.py` is used to preprocess raw dictionary data into the `frequency_dictionary_clean.txt` format, which is optimized for SymSpell.
  - **Testing**:
    - **`tests/test_layer1.py`**: Document the spelling-related tests within this file. Explain the purpose of fixtures like `spelling_checker` and what each test case (e.g., `test_dictionary_loading_success` verifies dictionary integrity, `test_correct_spelling_multiple_errors` checks batch correction, `test_spelling_correction_traceability` provides word-by-word change logs) validates.

---

### Team Member 3: Layer 1 - Grammar Checker

- **Objective**: To document the rule-based grammar correction module, its integration with CoreNLP, and the process for adding new rules.
- **Key Files**: `layer1/grammar_checker.py`, `data/grammar/`, `tests/test_layer1.py`, `tests/comprehensive_grammar_test.py`
- **Detailed Scope**:
  - **`grammar_checker.py`**:
    - **Class Overview**: Describe the `EnhancedGrammarChecker` class and its reliance on the Stanford CoreNLP server for advanced linguistic analysis (e.g., dependency parsing). Explain its role in identifying and correcting grammatical errors based on a predefined set of rules and a statistical N-gram model.
    - **Rule Engine**: Explain how the `_initialize_grammar_rules` method works and the structure of a `GrammarRule` (ID, pattern, correction, description). Provide a clear example of how to add a new regex-based rule, including common patterns for subject-verb agreement, article usage, modal verbs, and verb forms (gerunds vs. infinitives).
    - **CoreNLP Integration**: Detail how the `check_grammar_with_corenlp` method interacts with the CoreNLP parser to obtain linguistic annotations (POS tags, lemmas, dependency trees). Explain that these annotations are crucial for identifying complex grammatical structures that simple regex might miss.
    - **Readability Score**: Document the `calculate_readability` method. Explain that it computes readability scores (e.g., Flesch-Kincaid Grade Level, Flesch Reading Ease) to assess text complexity, and describe the interpretation of these scores.
    - **N-gram Model**: Explain the role of the N-gram model (loaded from `models/ngram_model.pkl`). Clarify that it is trained on a large corpus (e.g., Lang-8) and used for statistical validation of grammar corrections, helping to resolve ambiguities or rank potential corrections.
  - **Data Files**: Document the various grammar datasets in `data/grammar/` (FCE, CoNLL14, JFLEG, Lang-8). Explain their purpose in training the N-gram model and providing diverse examples for rule development.
  - **Testing**:
    - **`tests/test_layer1.py`**: Document the grammar-related tests. Explain what test cases like `test_check_grammar_rules` (for individual rule validation) and `test_parse_lang8` (for dataset parsing integrity) are verifying.
    - **`tests/comprehensive_grammar_test.py`**: Explain the purpose and usage of this script as a standalone, detailed test for all grammar rules. Describe how to run it, interpret its output (including pass/fail for each case and overall summary), and how to add new test cases.

---

### Team Member 4: Layer 2 - T5 AI Model

- **Objective**: To document the T5 AI-powered correction model, detailing its architecture, usage, and training process.
- **Key Files**: `layer2/ai_corrector.py`, `models/t5_grammar_spelling_model/`, `notebooks/data_preprocessing_t5.ipynb`, `tests/test_layer2.py`
- **Detailed Scope**:
  - **`ai_corrector.py`**:
    - **T5 Corrector**: Document the `T5GrammarCorrector` class, including how it loads the pre-trained T5 model.
  - **Models Directory**: Document the contents of the `models/t5_grammar_spelling_model/` directory, explaining `config.json`, `generation_config.json`, `model.safetensors`, `pytorch_model.bin`, and `tokenizer.json`.
  - **Notebooks**:
    - **`data_preprocessing_t5.ipynb`**: Document the data preprocessing steps for the T5 model. Include:
      - **Dataset Loading**: Explain the loading of the `jhu-clsp/jfleg` dataset and splitting into train/validation/test sets.
      - **Data Transformation**: Detail how grammar and spelling datasets are combined and transformed into `error` and `corrected` columns.
      - **Tokenization**: Explain the use of `AutoTokenizer.from_pretrained("t5-small")` and the tokenization function, including `max_length = 128`.
      - **Data Distribution**: Mention the final train/validation/test sizes and the distribution of 'spelling' vs 'grammar' error types.
  - **Testing**:
    - **`tests/test_layer2.py`**: Document the unit tests for the T5 model.

---

### Team Member 5: Layer 2 - Seq2Seq AI Model

- **Objective**: To document the Seq2Seq AI-powered correction model, detailing its architecture, usage, and data/training process.
- **Key Files**: `layer2/keras_models.py`, `models/seq2seq_model.keras`, `models/seq2seq_tokenizer.pkl`, `notebooks/data_preparation_seq2seq.ipynb`, `notebooks/newSeq2seq.ipynb`
- **Detailed Scope**:
  - **Data Preparation (`notebooks/data_preparation_seq2seq.ipynb`)**:
    - **Data Sources**: Explain the use of `missp.dat` for spelling error pairs and `jhu-clsp/jfleg` for grammar error pairs.
    - **Spelling Data Generation**: Detail how sentence templates (e.g., `I visited {} last year.`) are used to create synthetic spelling error/correction pairs from `missp.dat`.
    - **Grammar Data Extraction**: Explain how original sentences and their corrections are extracted from the JFLEG dataset.
    - **Combined Dataset**: Describe the process of combining spelling and grammar pairs and saving them to `combined_grammar_spelling_pairs.csv`.

  - **Model Development (`notebooks/newSeq2seq.ipynb`)**:
    - **Data Loading & Preprocessing**:
      - Explain loading `combined_grammar_spelling_pairs.csv`.
      - Detail the use of `Tokenizer` for vocabulary creation, including `num_words=10000`, `oov_token='<OOV>'`, and special tokens (`<pad>`, `<start>`, `<end>`).
      - Explain how `tokenizer.word_index` is adjusted to ensure `<pad>` has an index of 0.
      - Describe tokenization and sequence padding using `pad_sequences` with `padding='post'`.
      - Mention the derived `max_input_len` (e.g., 70) and `max_target_len` (e.g., 75).
      - Explain the preparation of `decoder_input_data` and `decoder_target_data`.
      - State the vocabulary size (e.g., 10001) and the sizes of training, validation, and test splits (e.g., 31300, 3912, 3913 samples respectively).
    - **Model Architecture**:
      - **Encoder**: Document the `Encoder` class (`tf.keras.Model`), explaining its `Embedding` layer and `LSTM` layer, and its role in converting input sequences to context vectors.
      - **BahdanauAttention**: Document the `BahdanauAttention` class (`tf.keras.layers.Layer`), explaining its mechanism for weighting encoder outputs to generate a context vector.
      - **Decoder**: Document the `Decoder` class (`tf.keras.Model`), detailing its `Embedding` layer, `LSTM` layer, integration with `BahdanauAttention`, and `Dense` output layer with `softmax` activation for generating target sequences.
      - **Seq2Seq Model**: Document the overarching `Seq2Seq` class (`tf.keras.Model`), describing how it combines the Encoder and Decoder for the end-to-end process.
    - **Hyperparameters**: List key hyperparameters used (e.g., `embedding_dim = 128`, `lstm_units = 256`).

  - **`keras_models.py`**:
    - **Seq2Seq Corrector**: Document the `Seq2SeqCorrector` class, explaining how it loads the Keras model (`seq2seq_model.keras`) and tokenizer (`seq2seq_tokenizer.pkl`) for inference.
  - **Models Directory**: Document the purpose of `seq2seq_model.keras` (the saved Keras model) and `seq2seq_tokenizer.pkl` (the saved tokenizer).

---

### Team Member 6: API, Backend, & Frontend

- **Objective**: To document the Flask application, including its endpoints and frontend components, explaining how they interact to provide the grammar and spelling correction service.
- **Key Files**: `app.py`, `templates/index.html`, `static/`, `tests/test_app_combined.py`, `tests/test_integration.py`
- **Detailed Scope**:
  - **API & Backend (`app.py`)**:
    - **Flask App Setup**: Explain the basic Flask application initialization and how it serves as the central orchestrator, loading and integrating the Layer 1 (spelling/grammar) and Layer 2 (AI models) components. Mention the lazy loading of models to optimize startup time.
    - **Endpoints**: Document the primary API endpoints:
      - **`/` (Root)**: Describe its role in serving the `index.html` frontend.
      - **`/check` (Quick Check)**: Detail its purpose for performing fast, rule-based spelling and grammar corrections. Explain that it primarily utilizes Layer 1.
      - **`/deepcheck` (Deep Check)**: Explain this endpoint's role in performing more comprehensive, AI-enhanced corrections, leveraging both Layer 1 and Layer 2 models. Highlight how Layer 2 is triggered based on Layer 1's confidence or specific error types.
      - **`/status`**: Document its purpose for providing real-time status updates on the server and loaded models.
    - **Request/Response Structure**: For `/check` and `/deepcheck`, specify the expected POST request format (e.g., `data={"text": "your sentence"}`) and the JSON response structure (e.g., `{"original_text": ..., "final_corrected_text": ..., "corrections": [...]}`).
  - **Frontend**:
    - **`templates/index.html`**: Document its role as the single-page application (SPA) entry point, describing its basic HTML structure for user input and displaying results.
    - **`static/js/script.js`**: Explain the client-side JavaScript logic. Detail how it captures user input, sends asynchronous `fetch` requests to the `/check` and `/deepcheck` API endpoints, and dynamically updates the HTML to display corrected text and highlight corrections.
    - **`static/css/style.css`**: Briefly describe its purpose in providing the visual styling for the web interface, ensuring a user-friendly and responsive design.
  - **Testing**:
    - **`tests/test_app_combined.py`**: Describe this script as an executable integration test for the live Flask server. Explain how it simulates API requests to `/check` and `/deepcheck` and verifies their responses.
    - **`tests/test_integration.py`**: Document the `pytest`-based integration tests. Explain that these tests use mocking to simulate backend components, allowing for isolated testing of API endpoint logic without requiring a running server. 
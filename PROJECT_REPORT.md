# COLLEGE MINI PROJECT REPORT

# Marathi News Classification Using Natural Language Processing

---

## Abstract
Automatic text classification forms the backbone of modern digital information retrieval and automated media organization. While extensive research has focused on English and high-resource European languages, low-resource and morphologically complex Indian languages like Marathi remain under-represented in production NLP systems. This project presents an end-to-end Machine Learning and Natural Language Processing architecture tailored specifically for Marathi news headline categorization across three target domains: **State (राज्य)**, **Entertainment (मनोरंजन)**, and **Sports (क्रीडा)**. The dataset is provided as separate training (`data/train.csv` containing 9,673 records) and validation (`data/valid.csv` containing 2,419 records) files. The proposed system features an Indic-specialized text cleaning pipeline preserving native Devanagari Unicode characters, Marathi stopword removal, sublinear Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering with unigram and bigram ranges across 6,884 features, and a comparative study of three fundamental classification models: Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machines (SVM). Experimental evaluation strictly on the unseen validation dataset demonstrates that the Calibrated Linear SVM achieves a validation accuracy of **83.75%** and a Weighted F1-Score of **83.20%**. The trained system is packaged and deployed using Flask as a production-grade web application alongside a standardized RESTful API and interactive CLI.

---

## Chapter 1 — Introduction

Natural Language Processing (NLP) is an interdisciplinary field of computer science, artificial intelligence, and computational linguistics that investigates computational techniques for analyzing and representing human languages. In an era marked by an unprecedented surge of digital media consumption, regional language journalism has witnessed substantial growth across India.

Marathi, spoken by over 83 million native speakers predominantly across the state of Maharashtra, ranks among the most spoken languages in India and the world. With leading Marathi publications—such as Loksatta, Sakal, Maharashtra Times, and Lokmat—publishing thousands of articles and news alerts daily across digital websites, mobile apps, and social feeds, the necessity for robust automated news categorization has become paramount.

Automated news classification involves assigning a pre-determined categorical label to an unannotated news headline or article. Accomplishing this for Marathi requires overcoming unique linguistic intricacies:
1. **Devanagari Script:** Marathi uses the Devanagari script, which contains 14 vowels, 36 consonants, and complex conjunct characters (जोडाक्षरे) along with dependent vowel signs (मात्रा).
2. **Agglutinative and Inflectional Morphology:** Marathi words frequently undergo morpho-syntactic inflection through case markers and postpositions directly attached to noun stems (विभक्ती प्रत्यय), expanding vocabulary surface forms.
3. **Domain Disambiguation:** Differentiating regional governance and state affairs from cultural entertainment and athletic achievements.

This project addresses these challenges by developing a complete computational pipeline that processes native Devanagari text, extracts domain-specific n-gram patterns, trains and benchmarks multiple supervised classifiers, and provides an accessible, intuitive interface for real-time inference.

---

## Chapter 2 — Problem Statement and Objectives

### 2.1 Problem Statement
Manual indexing, filtering, and routing of regional Marathi news articles across editorial workflows is slow, costly, and error-prone. Standard generic NLP tools often fail when directly applied to Marathi because they either rely on Latin transliteration or lack Indic-aware stopword dictionaries and tokenizers. Therefore, the problem is to design, implement, and benchmark an automated, end-to-end Natural Language Processing system capable of accurately predicting the domain category of native Marathi news headlines across 3 defined classes (**State**, **Entertainment**, **Sports**) with high confidence and minimal latency.

### 2.2 Objectives
The primary objectives of this project are:
1. **Linguistic Integrity:** Develop a Devanagari-native text preprocessing pipeline that performs NFKC Unicode normalization, noise cleaning, and Marathi stopword filtration without transliteration into English.
2. **Feature Representation:** Construct a rich, high-dimensional TF-IDF vector space capturing unigram and bigram contextual sequences with sublinear term-frequency scaling.
3. **Empirical Multi-Model Comparison:** Train on `data/train.csv` (9,673 samples) and evaluate on `data/valid.csv` (2,419 samples) across three candidate machine learning algorithms:
   * Multinomial Naive Bayes (generative baseline)
   * Logistic Regression (maximum-entropy discriminative baseline)
   * Linear Support Vector Machine (maximum-margin hyperplane classifier)
4. **Calibrated Confidence Scoring:** Integrate Platt sigmoid calibration (`CalibratedClassifierCV`) to output authentic posterior class probabilities.
5. **System Deployment & Verification:** Deliver a responsive, modern Flask web portal, standardized RESTful API (`POST /api/predict`), and command-line interface, rigorously tested across real-world test cases.

---

## Chapter 3 — Literature Survey

Text categorization has evolved through distinct paradigms in computational linguistics:

1. **Rule-Based & Keyword Approaches:**
   Early text classification relied heavily on handcrafted dictionaries, regular expressions, and syntactic grammars. While transparent, these systems proved fragile against linguistic diversity, informal phrasing, and colloquial vocabulary shifts.

2. **Statistical Bag-of-Words and TF-IDF:**
   Salton and Buckley (1988) formulated Term Frequency-Inverse Document Frequency (TF-IDF) weighting, establishing an enduring standard for information retrieval. Statistical models operating on TF-IDF matrices—notably Naive Bayes (McCallum & Nigam, 1998) and Support Vector Machines (Joachims, 1998)—demonstrated exceptional efficiency and accuracy across high-dimensional text datasets. Joachims demonstrated that text categorization problems are well suited for Linear SVMs due to sparse feature spaces, few irrelevant attributes, and dense document representations.

3. **Indic Language Computing & Marathi NLP:**
   Research in Indic NLP highlighted that European NLP assumptions break down when faced with the morphological complexity of Indo-Aryan and Dravidian languages (Bharati et al., 1995). Joshi et al. (2019) explored sentiment analysis and text classification in Hindi and Marathi, emphasizing the critical importance of Devanagari Unicode normalization and custom stopword exclusion.

4. **Deep Learning and Transformer Encoders:**
   Recent advancements include multilingual BERT (Devlin et al., 2019) and IndicBERT (Kakwani et al., 2020). While transformer models capture deep bidirectional semantics, their inference requires significant computational resources, GPU dependencies, and extended training time. For concise headline classification where latency and deployment simplicity are paramount, optimized n-gram TF-IDF paired with calibrated Linear SVM provides a highly competitive, lightweight, and cost-effective alternative.

---

## Chapter 4 — Existing System

### 4.1 Characteristics of Existing Systems
In many regional publishing houses and academic projects:
* **Manual Tagging:** Editors manually assign news categories upon publication.
* **Keyword Matching:** Simple string matching checks for fixed keywords.
* **Transliteration Pipelines:** Systems convert Marathi into Roman characters (English script) using phonetic engines before feeding them into standard English classifiers.

### 4.2 Limitations of the Existing System
* **High Human Overhead:** Manual curation does not scale with digital wire feeds and breaking news.
* **Brittleness of Keyword Logic:** If a headline discusses sports politics or celebrity athletes, naive keyword matching fails.
* **Loss of Linguistic Information in Transliteration:** Converting Devanagari to English phonetic script introduces ambiguity due to inconsistent spelling conventions.
* **Absence of Calibrated Probabilities:** Rule-based tools cannot measure predictive uncertainty.

---

## Chapter 5 — Proposed System

The proposed system replaces manual curation and naive keyword heuristics with a complete statistical NLP machine learning pipeline:

1. **Native Devanagari Pipeline:** Operates directly on UTF-8 Marathi Devanagari characters, maintaining morphological authenticity.
2. **Comprehensive Preprocessing:** Eliminates typographic noise, URLs, markup, punctuation, and common Marathi stopwords.
3. **N-Gram TF-IDF Feature Space:** Captures both standalone words and contextual bigrams across 6,884 features.
4. **Benchmarked Classifiers:** Evaluates three diverse statistical paradigms and automatically selects the top performer based on validation F1-score.
5. **Calibrated Probability Engine:** Employs sigmoid calibration so users receive genuine confidence estimates.
6. **Dual-Interface Deployment:** Supports web browser access via responsive Bootstrap 5 UI and machine-to-machine communication via JSON REST API.

---

## Chapter 6 — System Architecture

```text
                    Marathi News Headline
                              │
                              ▼
                     Text Preprocessing
            (Devanagari Normalization & Cleaning)
                              │
                              ▼
                    Marathi Tokenization
                              │
                              ▼
                      Stopword Removal
                              │
                              ▼
                           TF-IDF
               (Unigrams + Bigrams, Sublinear TF)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Naive Bayes    Logistic Regression  SVM
          (Acc: 82.80%)     (Acc: 82.76%)  (Acc: 83.75%)
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       Best Classifier
                     Linear SVM (83.20% F1)
                              │
                              ▼
                    Predicted News Category
                              │
                              ▼
                       Flask Web App
                              │
                              ▼
                       User / REST API
```

### Module Breakdown
* **Input Layer:** Receives UTF-8 raw Marathi news headlines through the Web UI, CLI, or REST API.
* **Preprocessing Module (`src/preprocessing.py`):** Performs character normalization, regex noise reduction, and stopword exclusion.
* **Feature Extraction Module (`src/train_model.py`):** Transforms clean text into sparse numerical vectors via TF-IDF.
* **Model Inference Layer (`src/predict.py`):** Loads saved artifacts (`models/news_classifier.pkl` and `models/tfidf_vectorizer.pkl`) and computes predictions with calibrated confidence.
* **Application Layer (`app.py`):** Exposes web endpoints (`/`, `/about`, `/model`, `/results`) and REST services (`/api/predict`, `/api/health`).

---

## Chapter 7 — Methodology

### 7.1 Dataset Construction and Validation
The project utilizes an authentic Marathi news dataset provided as separate training (`data/train.csv`) and validation (`data/valid.csv`) files:

* **Training Dataset (`data/train.csv`):** 9,673 samples
  * `State`: 6,035 samples (62.39%)
  * `Entertainment`: 2,644 samples (27.33%)
  * `Sports`: 994 samples (10.28%)
* **Validation Dataset (`data/valid.csv`):** 2,419 samples
  * `State`: 1,512 samples (62.51%)
  * `Entertainment`: 679 samples (28.07%)
  * `Sports`: 228 samples (9.43%)
* **Total Records:** 12,092 Marathi news headlines.

The training and validation sets were kept strictly separated—no dataset merging, synthetic generation, or random re-splitting was performed. The original CSV files were preserved without modification.

The data underwent strict validation:
* Missing value detection and elimination (0 missing entries found).
* String sanitation and whitespace trimming.
* Category integrity verification (State, Entertainment, Sports).

### 7.2 Indic NLP Preprocessing
1. **Unicode NFKC Normalization:** Converts distinct Unicode representations of composite Devanagari characters into canonical composed forms.
2. **Noise and Symbol Filtering:** Removes non-essential punctuation (English and Devanagari punctuation like danda `।` and double danda `॥`), web URLs, and email addresses.
3. **Devanagari Preservation:** Strictly retains Unicode range `\u0900-\u097F` along with relevant alphanumeric terms (e.g., 5G, AI, IPL).
4. **Stopword Filtering:** Excludes high-frequency functional words that do not carry domain specificity:
   $$\text{Stopwords} = \{ \text{आणि, व, किंवा, पण, परंतु, आहे, आहेत, होते, होती, केला, केली, ...} \}$$

### 7.3 TF-IDF Feature Extraction
Term Frequency-Inverse Document Frequency calculates the relative importance of n-gram $t$ within document $d$ against the corpus $D$:
$$\text{TF}(t, d) = \text{frequency of } t \text{ in } d$$
$$\text{Sublinear TF}(t, d) = 1 + \log(\text{TF}(t, d)) \quad \text{if } \text{TF}(t, d) > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{Sublinear TF}(t, d) \times \text{IDF}(t, D)$$

Features were configured with `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`, resulting in **6,884 unique n-gram features**.

---

## Chapter 8 — Algorithms

### 8.1 Multinomial Naive Bayes
A generative probabilistic classifier based on Bayes' theorem with the naive assumption of conditional feature independence given class $c$:
$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{V} P(x_i \mid c)$$
Using Laplace smoothing parameter $\alpha = 0.1$:
$$P(x_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha V}$$
* **Strengths:** Fast training, minimal computational requirements.
* **Limitations:** Independence assumption ignores word co-occurrences.

### 8.2 Logistic Regression
A discriminative generalized linear model optimizing the multinomial cross-entropy loss function with $L_2$ regularization:
$$P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{j=1}^{C} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
* **Strengths:** Produces calibrated probabilities, handles feature correlations well.
* **Limitations:** Decision boundary remains linear in the feature space.

### 8.3 Linear Support Vector Machine (Linear SVM)
A discriminative maximum-margin classifier that identifies the optimal hyperplane separating classes:
$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i$$
$$\text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
To obtain probabilistic confidence scores from uncalibrated hinge loss margins, the classifier was wrapped with **Platt Sigmoid Calibration** (`CalibratedClassifierCV` with 3-fold cross validation):
$$P(y = c \mid f(\mathbf{x})) = \frac{1}{1 + \exp(A f(\mathbf{x}) + B)}$$
* **Strengths:** Highly effective in high-dimensional sparse text vectors, resilient to overfitting.

---

## Chapter 9 — Implementation

### 9.1 Project Structure and Organization
The system follows a modular engineering structure:
* `src/preprocessing.py`: Standalone, reusable cleaning and tokenization logic.
* `src/train_model.py`: Execution pipeline automating loading, cleaning, vectorization, validation comparison, model selection, and visualization generation.
* `src/predict.py`: Encapsulated inference class `MarathiNewsPredictor` with CLI and programmatic hooks.
* `src/evaluate.py`: Matplotlib/Seaborn visualization scripts and Scikit-learn metric calculations.
* `app.py`: Flask web application managing template rendering, error dispatch, and REST routing.

### 9.2 Consistency Guarantee
By sharing `clean_marathi_text()` identically across training and real-time prediction, the system ensures zero preprocessing skew between the model's training representations and live inputs.

---

## Chapter 10 — Results and Analysis

### 10.1 Empirical Model Performance
The candidate classifiers were trained on `data/train.csv` (9,673 samples) and benchmarked against the identical 2,419 held-out validation headlines (`data/valid.csv`). The empirical metrics obtained are:

| Model | Validation Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score | Macro F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 82.80% | 82.77% | 82.80% | 81.97% | 76.03% |
| **Logistic Regression** | 82.76% | 82.80% | 82.76% | 81.92% | 75.36% |
| **Linear SVM (Calibrated)** | **83.75%** | **83.54%** | **83.75%** | **83.20%** | **77.74%** |

Linear SVM demonstrated the highest performance across all evaluation metrics, achieving a validation accuracy of **83.75%** and Weighted F1 of **83.20%**, and was automatically selected as the production model.

### 10.2 Per-Category Performance Breakdown
Detailed class-level performance of the selected Linear SVM model on the validation partition:

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Entertainment** | 0.82 | 0.68 | 0.75 | 679 |
| **Sports** | 0.80 | 0.61 | 0.69 | 228 |
| **State** | 0.85 | 0.94 | 0.89 | 1,512 |
| **Overall Accuracy** | — | — | **0.84 (83.75%)** | **2,419** |

### 10.3 Sample Headline Test Verification
The model was tested against representative Marathi headlines across each category:

1. **Entertainment:** `हृतिक करतोय बायोपिक 'सुपर 30'` → Predicted: **Entertainment** (Confidence: 51.65%)
2. **Entertainment:** `बॉलीवूड अभिनेता शाहरुख खानच्या नवीन चित्रपटाचे जोरदार प्रमोशन सुरू` → Predicted: **Entertainment** (Confidence: 97.86%)
3. **Sports:** `सचिन तेंडुलकर आणि विराट कोहलीने केली भारतीय क्रिकेट संघाची प्रशंसा, उत्कृष्ट फलंदाजी` → Predicted: **Sports** (Confidence: 93.66%)
4. **Sports:** `रोहित शर्माच्या नेतृत्वाखाली भारतीय संघाने जिंकला टी-20 विश्वचषक` → Predicted: **Sports** (Confidence: 86.60%)
5. **State:** `मुंबई पुणे महामार्गावर मुसळधार पावसामुळे दरड कोसळली, वाहतूक ठप्प` → Predicted: **State** (Confidence: 97.33%)
6. **State:** `मुख्यमंत्री यांनी घेतली राज्य मंत्रिमंडळाची महत्त्वाची बैठक, विविध विकासकामांना मंजुरी` → Predicted: **State** (Confidence: 99.29%)

---

## Chapter 11 — Applications

1. **Digital News Portals:** Automated section assignment for content management systems (CMS) in Marathi publishing agencies.
2. **Regional Media Aggregators:** Automated categorization and filtering for mobile news readers (e.g., Dailyhunt, Inshorts regional feeds).
3. **Media Monitoring & Public Sentiment Intelligence:** Tracking public discourse on governance, state infrastructure, and healthcare initiatives across Maharashtra.
4. **Smart Archives & Search Engines:** Tagging legacy journalistic archives for structured indexing and topic retrieval.

---

## Chapter 12 — Advantages and Limitations

### 12.1 Advantages
* **Native Devanagari Preservation:** Operates without lossy transliteration or phonetic degradation.
* **High Efficiency & Low Latency:** Sub-millisecond inference time enables high-throughput processing on standard CPU hardware.
* **Calibrated Confidence:** Yields genuine posterior probabilities rather than heuristic ranks.
* **Production-Grade Architecture:** Includes complete RESTful API, web dashboard, and test automation.

### 12.2 Limitations
* **Short Context Vulnerability:** Extreme brevity (less than 4 words) without domain keywords may reduce confidence.
* **Infrequent Vocabulary:** Rare morphological variants not encountered during training may produce out-of-vocabulary effects.
* **Single-Label Constraint:** Headlines touching multiple topics are constrained to the single dominant class.

---

## Chapter 13 — Future Scope

1. **Indic Transformer Models (IndicBERT / MuRIL):** Fine-tuning contextual transformer encoders to capture complex, long-range semantic dependencies and syntactic nuances.
2. **Multi-Label Categorization:** Extending the classification head to predict primary and secondary tags concurrently.
3. **Full Article Summarization:** Combining categorization with extractive text summarizers to generate regional news digests.
4. **Live RSS Scraping Integration:** Establishing streaming data pipelines directly connected to Marathi wire services.

---

## Chapter 14 — Conclusion

This project successfully developed, evaluated, and deployed a complete, end-to-end Machine Learning and Natural Language Processing system for Marathi News Headline Classification. Using an authentic dataset provided in separate training (`data/train.csv`, 9,673 samples) and validation (`data/valid.csv`, 2,419 samples) files across three target categories (**State**, **Entertainment**, **Sports**), the pipeline applies specialized Devanagari Unicode preprocessing, Marathi stopword removal, sublinear n-gram TF-IDF vectorization across 6,884 features, and a Calibrated Linear Support Vector Machine. The system achieved **83.75% validation accuracy** and **83.20% Weighted F1-score** on unseen validation samples. The project is delivered with an interactive Flask web application, standardized REST API endpoints, automated unit test suites, and empirical visual analytics.

---

## References

1. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.
2. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. *European Conference on Machine Learning (ECML)*, 137-142.
3. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 752, 41-48.
4. Bharati, A., Chaitanya, V., & Sangal, R. (1995). *Natural Language Processing: A Paninian Perspective*. Prentice-Hall of India.
5. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61-74.
6. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171-4186.
7. Kakwani, D., Kunchukuttan, A., Golla, S., Gokul, N. C., Bhattacharyya, A., et al. (2020). IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 4948-4961.
8. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

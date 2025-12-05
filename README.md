# User Review Analysis Project

## Overview
This project performs comprehensive sentiment analysis on restaurant reviews from the BDFoodSent-334k dataset, focusing on three key aspects: **Food**, **Ambiance**, and **Service**. The analysis employs multiple NLP models including BERT and TinyRoBERTa to extract aspect-specific sentiments and generate actionable insights.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Input Data](#input-data)
3. [Workflow](#workflow)
4. [Analysis Pipelines](#analysis-pipelines)
5. [Intermediate Results](#intermediate-results)
6. [Final Results](#final-results)
7. [Visualizations](#visualizations)
8. [Key Findings](#key-findings)
9. [Dependencies](#dependencies)
10. [Usage](#usage)

---

## Project Structure

```
user-review-analysis/
├── analysis_notebooks/          # Jupyter notebooks for analysis
│   ├── bert_aspect_question_answering_pipeline.ipynb
│   ├── tinyroberta_aspect_sentiment_pipeline.ipynb
│   ├── aspect_sentiment_diagnostic_report.ipynb
│   ├── multi_model_aspect_sentiment_analysis.ipynb
│   ├── review_dataset_exploration.ipynb
│   └── review_dataset_quick_checks.ipynb
├── data/                        # Input data directory
├── results/                     # Generated CSV outputs
│   ├── food_sentiment_bert_base_cased.csv
│   ├── food_sentiment_tinyroberta_squad2.csv
│   ├── ambiance_sentiment_bert_base_cased.csv
│   ├── ambiance_sentiment_tinyroberta_squad2.csv
│   ├── service_sentiment_bert_base_cased.csv
│   └── service_sentiment_tinyroberta_squad2.csv
├── imgs/                        # Visualization outputs
│   ├── sentiment_label_distribution/
│   ├── phrase_analysis/
│   ├── aspect_comparison/
│   ├── correlations/
│   └── text_span_analysis/
└── docs/                        # Documentation
    └── analysis_summary.md
```

---

## Input Data

### Primary Dataset: BDFoodSent-334k.csv
- **Source**: Bangladesh Food Sentiment Dataset
- **Size**: ~334,000 reviews
- **Columns**:
  - `text`: Review text content
  - `name`: Restaurant identifier
  - `city`: Restaurant location
  - Additional metadata fields

### Data Characteristics:
- Multi-lingual reviews (English-heavy filtering applied)
- Varying review lengths (3+ words minimum after filtering)
- Covers multiple restaurants across different cities
- Contains unstructured text requiring NLP processing

---

## Workflow

### Phase 1: Data Exploration & Preprocessing
1. **Initial Dataset Loading** (`review_dataset_exploration.ipynb`)
   - Load BDFoodSent-334k.csv
   - Examine column structure and data types
   - Identify missing values and data quality issues
   - Generate descriptive statistics

2. **Data Quality Checks** (`review_dataset_quick_checks.ipynb`)
   - Validate review text completeness
   - Check for duplicate entries
   - Assess language distribution

### Phase 2: Aspect Extraction
3. **BERT-based Question Answering** (`bert_aspect_question_answering_pipeline.ipynb`)
   - **Model**: `deepset/bert-base-cased-squad2`
   - **Process**:
     1. Define aspect-specific questions:
        - Food: "What is the opinion about the taste, quantity or food quality?"
        - Service: "What is the feedback on the service or delivery?"
        - Ambiance: "What is the description of the restaurant's atmosphere or environment?"
     2. Extract answers for each review × aspect combination
     3. Calculate answer confidence scores
     4. Compute cosine similarity between questions and answers
     5. Apply text filters (minimum word count, English ratio)
     6. Select top 10% highest-scoring reviews per aspect
   
   - **Output**: 
     - `food_sentiment_bert_base_cased.csv`
     - `ambiance_sentiment_bert_base_cased.csv`
     - `service_sentiment_bert_base_cased.csv`

4. **TinyRoBERTa Sentiment Pipeline** (`tinyroberta_aspect_sentiment_pipeline.ipynb`)
   - **Model**: Fine-tuned TinyRoBERTa on SQuAD2
   - **Process**:
     1. Load pre-filtered aspect-specific reviews
     2. Extract sentiment-bearing text spans
     3. Apply sentiment classification (1-5 stars)
     4. Calculate sentiment confidence scores
     5. Assign sentiment labels
   
   - **Output**:
     - `food_sentiment_tinyroberta_squad2.csv`
     - `ambiance_sentiment_tinyroberta_squad2.csv`
     - `service_sentiment_tinyroberta_squad2.csv`

### Phase 3: Sentiment Analysis
5. **Multi-Model Comparison** (`multi_model_aspect_sentiment_analysis.ipynb`)
   - Compare BERT vs TinyRoBERTa predictions
   - Analyze model agreement rates
   - Identify systematic differences in span extraction
   - Evaluate confidence score distributions

6. **Diagnostic Reporting** (`aspect_sentiment_diagnostic_report.ipynb`)
   - Generate comprehensive sentiment statistics
   - Create visualizations for each aspect
   - Perform phrase-level analysis
   - Extract top positive/negative phrases
   - Build aspect comparison metrics

---

## Analysis Pipelines

### Pipeline 1: BERT Question-Answering
**Input**: Raw reviews (BDFoodSent-334k.csv)  
**Steps**:
1. Question-Answer extraction for 3 aspects
2. Cosine similarity scoring
3. Text filtering (length, English ratio)
4. Top 10% selection per aspect
5. Sentiment scoring with `nlptown/bert-base-multilingual-uncased-sentiment`

**Output**: 3 CSV files with columns:
- `review`, `restaurant_id`, `city`, `question`, `answer`, `confidence`, `cosine_sim`, `sentiment`

### Pipeline 2: TinyRoBERTa Sentiment Classification
**Input**: BERT-extracted aspect reviews  
**Steps**:
1. Load aspect-specific reviews
2. Apply TinyRoBERTa for sentiment extraction
3. Generate 5-star sentiment labels
4. Calculate sentiment confidence scores

**Output**: 3 CSV files with columns:
- `text`, `name`, `city`, `Aspect_Answer`, `Aspect_Similarity_Score`, `Sentiment_Label`, `Sentiment_Confidence`

### Pipeline 3: Comparative Analysis
**Input**: Both BERT and TinyRoBERTa outputs  
**Steps**:
1. Merge predictions on review-restaurant pairs
2. Calculate exact match agreement rates
3. Analyze sentiment polarity alignment
4. Compare extracted phrase lengths
5. Generate model behavior statistics

---

## Intermediate Results

### 1. QA Extraction Results
- **Total extracted answers**: ~1 million (334k reviews × 3 aspects)
- **Confidence distribution**: Mean ~0.65, showing reasonable extraction quality
- **Cosine similarity scores**: Mean ~0.58, indicating moderate semantic alignment

### 2. Filtered Subsets (Top 10% per Aspect)
- **Food**: ~21,737 reviews
- **Ambiance**: ~21,737 reviews  
- **Service**: ~21,737 reviews

### 3. Sentiment Score Distribution
- **BERT sentiment**: Float values representing review positivity
- **TinyRoBERTa labels**: Categorical (1-5 stars)

### 4. Model Agreement Metrics
- **Food aspect exact match**: ~44%
- **Ambiance aspect exact match**: ~12% (lowest due to aspect leakage)
- **Service aspect exact match**: ~40%

---

## Final Results

### Results Files (CSV Format)

Each CSV contains sentiment predictions with the following schema:

#### BERT Results
| Column | Description |
|--------|-------------|
| `review` | Original review text |
| `restaurant_id` | Restaurant identifier |
| `city` | Restaurant location |
| `question` | Aspect question used |
| `answer` | Extracted aspect-specific phrase |
| `confidence` | QA model confidence (0-1) |
| `cosine_sim` | Semantic similarity score (0-1) |
| `sentiment` | Sentiment score (float) |

#### TinyRoBERTa Results
| Column | Description |
|--------|-------------|
| `text` | Original review text |
| `name` | Restaurant identifier |
| `city` | Restaurant location |
| `Aspect_Answer` | Extracted sentiment phrase |
| `Aspect_Similarity_Score` | Similarity score |
| `Sentiment_Label` | Star rating (1-5 stars) |
| `Sentiment_Confidence` | Prediction confidence (0-1) |

### Quantitative Findings

#### Food Aspect
- **Average rating**: 2.80 (below neutral)
- **1-star reviews**: 21.6%
- **5-star reviews**: 14.1%
- **Net sentiment**: -7.5%
- **Key insight**: Food quality is the weakest aspect with consistent negative feedback

#### Ambiance Aspect
- **Average rating**: 3.26 (above neutral)
- **1-star reviews**: 21.6%
- **5-star reviews**: 32.0%
- **Net sentiment**: +10.4%
- **Key insight**: Only aspect with positive sentiment; customers appreciate atmosphere

#### Service Aspect
- **Average rating**: 2.84 (below neutral)
- **1-star reviews**: 31.2% (highest across all aspects)
- **5-star reviews**: 22.9%
- **Net sentiment**: -8.3%
- **Key insight**: Most polarized aspect; delivery issues dominate complaints

---

## Visualizations

### Generated Visualizations (PNG files)

#### 1. Sentiment Label Distribution
**Location**: `imgs/sentiment_label_distribution/`
- `food_sentiment_label_distribution.png`
- `ambiance_sentiment_label_distribution.png`
- `service_sentiment_label_distribution.png`

**Shows**: Bar charts with count and percentage of each star rating (1-5)

#### 2. Phrase Analysis
**Location**: `imgs/phrase_analysis/`
- `food_phrase_analysis.png`
- `ambiance_phrase_analysis.png`
- `service_phrase_analysis.png`

**Shows**: 
- Top 10 positive phrases (bar chart + word cloud)
- Top 10 negative phrases (bar chart + word cloud)

#### 3. Aspect Comparison
**Location**: `imgs/aspect_comparison/`
- `aspect_mean_std.png` - Average ratings with standard deviation
- `aspect_star_extremes.png` - 1-star vs 5-star distribution
- `aspect_radar_chart.png` - Multi-dimensional aspect comparison

#### 4. Correlation Analysis
**Location**: `imgs/correlations/`
- `food_correlation_matrix.png`
- `ambiance_correlation_matrix.png`
- `service_correlation_matrix.png`

**Shows**: Correlation between sentiment scores and various features

#### 5. Text Span Analysis
**Location**: `imgs/text_span_analysis/`
- Review length vs sentiment scatter plots
- Span length comparisons between models
- Distribution analyses per aspect

---

## Key Findings

### 1. Aspect Performance Ranking
1. **Ambiance** ✅ - Only positive net sentiment (+10.4%)
2. **Service** ⚠️ - Highly polarized, net negative (-8.3%)
3. **Food** ❌ - Weakest performer, net negative (-7.5%)

### 2. Top Positive Phrases by Aspect

**Food:**
- "food quality" (146 mentions)
- "taste" (102 mentions)
- "great taste" (99 mentions)
- "excellent taste" (64 mentions)

**Service:**
- "excellent" (312 mentions)
- "quality" (270 mentions)
- "delivery" (119 mentions)
- "good quality" (102 mentions)

**Ambiance:**
- Note: Significant aspect leakage detected (food-related phrases dominate)
- Phrase-level analysis unreliable for this aspect

### 3. Top Negative Phrases by Aspect

**Food:**
- "food quality" (246 mentions)
- "taste" (179 mentions)
- "bad taste" (156 mentions)
- "no taste" (138 mentions)

**Service:**
- "delivery" (483 mentions - 4x higher than positive)
- "quality" (214 mentions)
- "refund" (85 mentions)
- "poor service" (77 mentions)

### 4. Model Behavior Insights

**BERT:**
- Extracts short, sentiment-focused phrases
- More stable and interpretable
- Better for aspect-level sentiment analysis

**TinyRoBERTa:**
- Extracts longer, contextual spans
- Behaves like QA span retriever
- Higher disagreement on phrase boundaries
- More prone to aspect leakage

### 5. Critical Business Insights

**Immediate Action Required:**
1. **Service/Delivery**: 31.2% negative reviews - highest complaint rate
   - Delivery is mentioned 4x more in negative vs positive reviews
   - Focus on delivery speed and reliability

2. **Food Taste**: Consistency issues evident
   - Negative taste mentions 2x higher than positive
   - "No taste" / "bad taste" are dominant complaints

**Strength to Maintain:**
3. **Ambiance**: 32% five-star reviews
   - Only aspect with net positive sentiment
   - Leverage in marketing and customer experience design

---

## Dependencies

### Core Libraries
```
pandas>=1.3.0
numpy>=1.20.0
torch>=1.9.0
transformers>=4.11.0
sentence-transformers>=2.0.0
nltk>=3.6.0
```

### Visualization
```
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
wordcloud>=1.8.0
```

### Utilities
```
tqdm>=4.62.0
scipy>=1.7.0
pathlib
```

---

## Usage

### 1. Environment Setup
```bash
# Install dependencies
pip install -q pandas torch nltk sentence-transformers transformers tqdm matplotlib seaborn plotly wordcloud scipy

# Download NLTK data
python -c "import nltk; nltk.download('words')"
```

### 2. Data Preparation
Place the input dataset at:
```
review_datasets/BDFoodSent-334k.csv
```

### 3. Run Analysis Pipeline

**Step 1: Initial exploration**
```bash
jupyter notebook analysis_notebooks/review_dataset_exploration.ipynb
```

**Step 2: BERT aspect extraction**
```bash
jupyter notebook analysis_notebooks/bert_aspect_question_answering_pipeline.ipynb
```

**Step 3: TinyRoBERTa sentiment analysis**
```bash
jupyter notebook analysis_notebooks/tinyroberta_aspect_sentiment_pipeline.ipynb
```

**Step 4: Generate diagnostic report**
```bash
jupyter notebook analysis_notebooks/aspect_sentiment_diagnostic_report.ipynb
```

**Step 5: Multi-model comparison**
```bash
jupyter notebook analysis_notebooks/multi_model_aspect_sentiment_analysis.ipynb
```

### 4. View Results
- CSV outputs: `results/`
- Visualizations: `imgs/`
- Summary report: `docs/analysis_summary.md`

---

## Workflow Diagram

```
┌─────────────────────────────────────┐
│   BDFoodSent-334k.csv (Input)      │
│         ~334k reviews               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  BERT Question-Answering Pipeline   │
│  - Extract aspect-specific answers  │
│  - Calculate cosine similarity      │
│  - Filter & select top 10%          │
│  - Generate sentiment scores        │
└──────────────┬──────────────────────┘
               │
               ├──► food_sentiment_bert_base_cased.csv
               ├──► ambiance_sentiment_bert_base_cased.csv
               └──► service_sentiment_bert_base_cased.csv
               │
               ▼
┌─────────────────────────────────────┐
│  TinyRoBERTa Sentiment Pipeline     │
│  - Load aspect reviews              │
│  - Extract sentiment spans          │
│  - Classify 1-5 star ratings        │
│  - Calculate confidence scores      │
└──────────────┬──────────────────────┘
               │
               ├──► food_sentiment_tinyroberta_squad2.csv
               ├──► ambiance_sentiment_tinyroberta_squad2.csv
               └──► service_sentiment_tinyroberta_squad2.csv
               │
               ▼
┌─────────────────────────────────────┐
│    Diagnostic & Comparison          │
│  - Model agreement analysis         │
│  - Sentiment distribution stats     │
│  - Phrase extraction & ranking      │
│  - Generate visualizations          │
└──────────────┬──────────────────────┘
               │
               ├──► imgs/sentiment_label_distribution/
               ├──► imgs/phrase_analysis/
               ├──► imgs/aspect_comparison/
               └──► docs/analysis_summary.md
```

---

## Model Details

### 1. BERT Base Cased (SQuAD2)
- **Model**: `deepset/bert-base-cased-squad2`
- **Task**: Question Answering
- **Input**: Review text + aspect question
- **Output**: Answer span + confidence score
- **Strengths**: Precise span extraction, high accuracy
- **Limitations**: May miss contextual nuances

### 2. BERT Multilingual Sentiment
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Task**: Sentiment Classification
- **Input**: Review text
- **Output**: Sentiment score (float)
- **Strengths**: Multi-language support, robust scoring
- **Limitations**: Requires clean, coherent text

### 3. TinyRoBERTa (SQuAD2)
- **Model**: TinyRoBERTa fine-tuned on SQuAD2
- **Task**: Extractive QA + Sentiment
- **Input**: Review text + aspect question
- **Output**: Answer span + 5-star label + confidence
- **Strengths**: Longer context extraction
- **Limitations**: Aspect leakage, less precise boundaries

### 4. Sentence Transformer
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Task**: Semantic Similarity
- **Input**: Question-answer pairs
- **Output**: Cosine similarity score
- **Purpose**: Quality filtering of extracted aspects

---

## Future Enhancements

1. **Model Improvements**
   - Implement aspect-specific fine-tuning
   - Address ambiance aspect leakage
   - Ensemble methods for better accuracy

2. **Analysis Extensions**
   - Temporal trend analysis
   - Geographic sentiment patterns
   - Restaurant-level recommendations

3. **Automation**
   - End-to-end pipeline script
   - Real-time review processing
   - Automated report generation

4. **Deployment**
   - API for sentiment prediction
   - Dashboard for visualization
   - Integration with review platforms

---

## Citation

If you use this project, please cite:
```
User Review Analysis Project
GitHub: https://github.com/manasvin24/user-review-analysis
Dataset: BDFoodSent-334k
```

---

## License

This project follows the license terms of the source dataset and included models.

---

## Contact

For questions or contributions, please open an issue on the GitHub repository.

**Last Updated**: December 4, 2025

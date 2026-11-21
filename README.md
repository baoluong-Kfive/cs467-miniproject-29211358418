# CS467 Mini-Project: Book Summary and Reader-Taste Recommender (MP-A02)

[![Course](https://img.shields.io/badge/Course-CS467%20Advanced%20AI-blue.svg)](https://duytan.edu.vn/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Information & Student Details

| Field | Details |
|---|---|
| **Student Name** | Lương Gia Bảo |
| **Student ID / Section** | 29211358418 \| CS467A (No. 02) |
| **Instructor** | Puri Vikram |
| **Assigned Project Code** | MP-A02 |
| **Assigned Project Title** | Book Summary and Reader-Taste Recommender |
| **Application Domain** | Online Bookstore and Digital Library |
| **Required GitHub Repo Name** | `cs467-miniproject-29211358418` |
| **University** | Duy Tan University — School of Computer Science |

---

## 2. Problem Statement

In an online bookstore or university digital library setting, readers frequently encounter information overload when navigating extensive catalogs. Readers typically discover new titles through two distinct behaviors:
1. **Thematic affinity**: Wanting recommendations reflecting the narrative style, subjects, and themes of books they previously enjoyed.
2. **Social consensus & peer tastes**: Relying on feedback and high ratings from other readers who share similar reading preferences.

This project designs and implements an intelligent hybrid recommendation system integrating:
- **Natural Language Processing (NLP)** over book descriptions, blurbs, and keyword metadata to build granular reader taste profiles and handle ad-hoc semantic search (including new reader cold-start).
- **User-Based Collaborative Filtering (CF)** to capture collective reader behaviors and predict ratings via co-rated books.
- **A Popularity Baseline** to benchmark recommendation quality and handle sparse cold-start scenarios.

---

## 3. Dataset Description

The dataset was curated specifically for a university reading club with varied genres, realistic sparse interactions, and descriptive text features.

```
data/
├── items.csv      # 42 books across diverse categories with text descriptions and keywords
└── ratings.csv    # 197 reader ratings and qualitative notes from 14 active readers
```

### 3.1. Items Dataset (`data/items.csv`)
- **Quantity**: 42 book titles (exceeds the 35–50 requirement).
- **Attributes**:
  - `item_id`: Unique identifier (e.g., `B001`–`B042`).
  - `title`: Book title.
  - `author`: Author name.
  - `category`: Genre category (e.g., *Historical Mystery*, *Science Fiction*, *Fantasy*, *Self-Help*, *Literary Fiction*, *Non-Fiction*, *Business & Finance*).
  - `blurb`: 1–2 sentence synopsis highlighting thematic plot points and style.
  - `keywords`: Tagged keywords emphasizing themes, time periods, and stylistic motifs.

### 3.2. User Interactions Dataset (`data/ratings.csv`)
- **Quantity**: 14 readers (`U01`–`U14`) with **197 total interactions**.
- **Attributes**:
  - `user_id`: Reader identifier (`U01`–`U14`).
  - `item_id`: Rated book identifier.
  - `rating`: Explicit integer rating from 1 to 5 stars.
  - `reader_note`: Qualitative short review / personal reader impressions.

### 3.3. Interaction Matrix & Sparsity
- **Total Possible Pairs**: $14 \text{ users} \times 42 \text{ items} = 588$ entries.
- **Observed Ratings**: 197 interactions.
- **Interaction Matrix Sparsity**:
  $$\text{Sparsity} = 1.0 - \left(\frac{197}{14 \times 42}\right) = 66.50\%$$
- **Reader Engagement**: Each user has rated between 11 and 15 books (well above the required 4–6 threshold), ensuring a realistic and balanced sparse interaction matrix.

---

## 4. NLP Pipeline and Recommendation Algorithms

The core logic is structured across `src/nlp_pipeline.py` and `src/recommender.py`.

```
src/
├── nlp_pipeline.py   # Text preprocessing, TF-IDF vectorization, user profile modeling
├── recommender.py    # Popularity, NLP Content-Based, Collaborative Filtering, Hybrid models
├── evaluate.py       # Hold-out split validation, Precision@5, RMSE, MAE, plotting
└── main.py           # Command-line interface and entry point
```

### 4.1. NLP Preprocessing and Feature Extraction (`nlp_pipeline.py`)
1. **Text Normalization**:
   - Lowercasing, punctuation stripping via regex (`[^a-z0-9\s\-]`), and whitespace collapsing.
   - Domain-specific stop-word removal (merging `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` with noise terms such as *novel, book, story, character, read*).
2. **Corpus Construction**:
   - Synthesizes `blurb`, `keywords` (weighted $2\times$), `category` (weighted $2\times$), and `title` into a unified document vector for each item.
3. **TF-IDF Vectorization**:
   - `TfidfVectorizer(max_features=1500, ngram_range=(1, 2), sublinear_tf=True)`.
4. **Weighted Reader Taste Profile**:
   - For user $u$, their profile vector $\vec{P}_u$ is constructed as a rating-weighted combination of rated book TF-IDF vectors $\vec{v}_i$:
     $$w_{u,i} = r_{u,i} - \text{threshold\_shift} \quad (\text{threshold\_shift} = 3.0)$$
     $$\vec{P}_u = \frac{\sum_{i \in I_u} \max(w_{u,i}, 0) \cdot \vec{v}_i}{\left\|\sum_{i \in I_u} \max(w_{u,i}, 0) \cdot \vec{v}_i\right\|_2}$$
   - Highly rated books ($\ge 4$) pull the profile toward preferred themes; disliked items are suppressed.
5. **Top Matching Keyword Extraction**:
   - Computes element-wise overlap scores $q_k \cdot d_k$ between the query vector and book vector to extract the most influential matched terms for explainable recommendations.

### 4.2. Recommendation Approaches (`recommender.py`)

#### 1. Popularity Baseline
- Employs a Bayesian weighted average rating with a minimum rating threshold ($m = 2$):
  $$\text{Score}(i) = \left(\frac{v_i}{v_i + m}\right) R_i + \left(\frac{m}{v_i + m}\right) C$$
  where $v_i$ is rating count, $R_i$ is item average rating, and $C$ is catalog global average rating ($C \approx 3.5$).

#### 2. NLP Content-Based Recommender
- Uses Cosine Similarity in the TF-IDF feature space:
  $$\text{Sim}(\vec{P}_u, \vec{v}_i) = \frac{\vec{P}_u \cdot \vec{v}_i}{\|\vec{P}_u\| \|\vec{v}_i\|}$$
- Supports:
  - **User Taste Profile Recommendation**: Recommending unread books closest to the reader's aggregated profile.
  - **Free-Text Search & Cold-Start**: Vectorizing arbitrary user input on-the-fly and ranking items.

#### 3. User-Based Collaborative Filtering (Pearson / Cosine)
- Computes pairwise user similarity $s(u, v)$ over co-rated books with significance shrinkage:
  $$s(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}} \times \left(\frac{|I_{uv}|}{|I_{uv}| + \beta}\right)$$
- Predicts missing ratings using mean-centered neighbor deviations ($k=5$ neighbors):
  $$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_i(u)} s(u, v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N_i(u)} |s(u, v)|}$$

#### 4. Hybrid Recommender
- Linearly blends normalized scores from Content-Based and Collaborative Filtering:
  $$\text{Score}_{\text{Hybrid}}(u, i) = \alpha \cdot \text{Sim}_{\text{CB}}(u, i) + (1 - \alpha) \cdot \left(\frac{\hat{r}_{u,i} - 1}{4}\right) \quad (\alpha = 0.55)$$

---

## 5. Installation and Run Commands

### 5.1. Requirements & Setup
Clone the repository and install required packages:

```bash
# Clone the repository
git clone https://github.com/baolug14/cs467-miniproject-29211358418.git
cd cs467-miniproject-29211358418

# (Optional) Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2. Running the Application

#### A. Run Full Offline Evaluation & Generate Charts (Default)
```bash
python src/main.py
```
This executes the train/test split, prints the evaluation comparison table, runs all 3 required test cases, saves `results/sample_outputs.txt`, and generates charts in `results/`.

#### B. Recommend Books for an Existing Reader
```bash
# Hybrid recommender (default)
python src/main.py --user U01

# Compare by specific method
python src/main.py --user U01 --method content
python src/main.py --user U01 --method cf
python src/main.py --user U01 --method popularity
```

#### C. Free-Text Search & Cold-Start Preference Query
```bash
python src/main.py --query "historical mystery set in Asia with detective"
python src/main.py --query "I enjoy cozy stories about warm neighborhood cafes, nature, and gentle everyday life."
```

#### D. Interactive Command-Line Interface
```bash
python src/main.py --interactive
```
Example interactive session:
```text
> user U01 hybrid
> query cyberpunk ai thriller
> user U03 content
> exit
```

---

## 6. Sample Outputs for Required Test Cases

### Test Case 1: Existing Reader Recommendations (`U01`, 15 Rated Books)
`U01` is an avid reader who loves **Asian Historical Mysteries** (`B001`, `B002`, `B003`, `B004`, `B021`, `B037` rated 4–5★) and dislikes heavy tech sci-fi (`B006` rated 2★).

```text
[1A] NLP Content-Based Reader Profile Top-5:
  1. [B032] Voices from the Silk Looms by Ananya Sharma (History) | Score: 0.0518 | Keywords: asia, history, silk
  2. [B017] Echoes of the Red Dunes by Amara Okafor (Science Fiction) | Score: 0.0369 | Keywords: mystery, fiction
  3. [B042] A Quiet Garden on the Hill by Elena Moreau (Literary Fiction) | Score: 0.0301 | Keywords: literary, literary fiction, gentle, cozy
  4. [B034] The Sunlit Greenhouse by Chloe Dubois (Literary Fiction) | Score: 0.0293 | Keywords: literary, literary fiction, cozy, gentle
  5. [B041] The Cozy Bookshop on Elm Street by Priya Sharma (Contemporary Fiction) | Score: 0.0271 | Keywords: cozy, gentle, life, everyday life

[1B] User-Based Collaborative Filtering (Pearson) Top-5:
  1. [B034] The Sunlit Greenhouse by Chloe Dubois (Literary Fiction) | Score: 5.0000 | Keywords: literary, literary fiction, cozy, gentle
  2. [B035] Quantum Horizons by Arjun Patel (Science & Tech) | Score: 4.7203 | Keywords: quantum, tech, science tech, science quantum
  3. [B039] Strategic Patience by Owen Gallagher (Business & Finance) | Score: 4.5900 | Keywords: patience, markets, finance, patience markets
  4. [B017] Echoes of the Red Dunes by Amara Okafor (Science Fiction) | Score: 4.4300 | Keywords: mystery, fiction
  5. [B029] The Last Observatory by Fatima Al-Zahra (Science Fiction) | Score: 4.4300 | Keywords: high, fiction

[1C] Hybrid (NLP + Collaborative Filtering) Top-5:
  1. [B034] The Sunlit Greenhouse by Chloe Dubois (Literary Fiction) | Score: 0.4661 | Keywords: literary, literary fiction, cozy, gentle
  2. [B035] Quantum Horizons by Arjun Patel (Science & Tech) | Score: 0.4185 | Keywords: quantum, tech, science tech, science quantum
  3. [B032] Voices from the Silk Looms by Ananya Sharma (History) | Score: 0.4072 | Keywords: asia, history, silk
  4. [B017] Echoes of the Red Dunes by Amara Okafor (Science Fiction) | Score: 0.4062 | Keywords: mystery, fiction
  5. [B039] Strategic Patience by Owen Gallagher (Business & Finance) | Score: 0.4039 | Keywords: patience, markets, finance, patience markets
```

### Test Case 2: Free-Text Search
**Query**: `'historical mystery set in Asia with detective'`

```text
  1. [B005] The Monsoon Detective by Arjun Nair (Historical Mystery) | Score: 0.4840 | Keywords: detective, historical, historical mystery, mystery
  2. [B037] Chronicle of the Amber Seal by Min-Seo Kang (Historical Mystery) | Score: 0.4550 | Keywords: historical, historical mystery, detective, mystery
  3. [B022] The Jade Pagoda Riddle by Linnea Bergstrom (Historical Mystery) | Score: 0.4453 | Keywords: historical, historical mystery, detective, mystery
  4. [B002] Shadows Over Shanghai 1932 by Mei-Ling Zhou (Historical Mystery) | Score: 0.4368 | Keywords: historical, historical mystery, detective, mystery
  5. [B004] Murder at the Forbidden Pavilion by Wei-Ming Chen (Historical Mystery) | Score: 0.4196 | Keywords: historical, historical mystery, detective, mystery
```
*Note*: Top 5 matches all belong strictly to the target genre and setting, with explicit matching keywords displayed.

### Test Case 3: Cold-Start New Reader Test
**Input**: `'I enjoy cozy stories about warm neighborhood cafes, nature, and gentle everyday life.'`

```text
  1. [B041] The Cozy Bookshop on Elm Street by Priya Sharma (Contemporary Fiction) | Score: 0.4138 | Keywords: everyday life, life, neighborhood, cozy
  2. [B042] A Quiet Garden on the Hill by Elena Moreau (Literary Fiction) | Score: 0.3959 | Keywords: nature gentle, nature, neighborhood, gentle
  3. [B014] The Midnight Tea House by Haruka Sato (Literary Fiction) | Score: 0.3411 | Keywords: cozy warm, warm, gentle, cozy
  4. [B034] The Sunlit Greenhouse by Chloe Dubois (Literary Fiction) | Score: 0.2486 | Keywords: gentle everyday, neighborhood, everyday, cozy
  5. [B027] The Autumn Orchard by Camille Laurent (Literary Fiction) | Score: 0.0589 | Keywords: nature
```
*Note*: For a brand-new user with zero interaction history, the system successfully extracts semantic cues and recommends cozy, slice-of-life titles with positive similarity scores.

---

## 7. Evaluation Results and Visualizations

### 7.1. Offline Evaluation Protocol
- **Split Strategy**: Per-user holdout validation hiding $n=2$ highly-rated books ($\text{rating} \ge 4.0$) per user.
- **Metrics**:
  - **Precision@5**: Ratio of relevant items in the top-5 recommendations.
  - **Recall@5**: Proportion of held-out relevant items captured.
  - **HitRate@5**: Proportion of users who received at least one relevant recommendation.
  - **RMSE / MAE**: Rating prediction error for collaborative filtering models.

### 7.2. Performance Comparison Table

| Model | Precision@5 | Recall@5 | HitRate@5 | RMSE | MAE |
|---|:---:|:---:|:---:|:---:|:---:|
| **Popularity Baseline** | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| **NLP Content-Based** | **0.3571** | **0.8929** | **1.0000** | N/A | N/A |
| **User-Based CF (Cosine)** | 0.0286 | 0.0714 | 0.1429 | 1.3490 | 1.0435 |
| **User-Based CF (Pearson)** | 0.0429 | 0.1071 | 0.2143 | **1.0968** | **0.8772** |
| **Hybrid (NLP + Pearson CF)** | 0.2000 | 0.5000 | 0.7143 | N/A | N/A |

### 7.3. Generated Visualizations
The evaluation script automatically produces two charts saved in the `results/` folder:

1. **Dataset Distribution Chart (`results/dataset_distribution.png`)**:
   - Left panel: Reader rating distribution (1 to 5 stars).
   - Right panel: Books per genre category in the catalog.

2. **Model Comparison Chart (`results/model_comparison.png`)**:
   - Grouped bar chart comparing Precision@5, Recall@5, and HitRate@5 across all five recommendation strategies.

---

## 8. Reflection & Limitations

### 8.1. What Worked Well
1. **Strong NLP Content-Based Performance**: TF-IDF combined with n-grams and domain keyword weighting achieved an outstanding **HitRate@5 of 100%** and **Recall@5 of 89.29%**. Readers with distinct genre preferences received very accurate matches.
2. **Cold-Start Resilience**: The NLP model gracefully supports free-text query transformation, enabling instant, high-quality recommendations for new readers who only provide a single descriptive sentence.
3. **Keyword Explainability**: Matched TF-IDF keywords provide transparent, human-readable explanations for why each book was suggested.
4. **Hybrid Balance**: Pearson CF reduced prediction error ($\text{RMSE} = 1.0968$), and combining it with Content-Based scores provides serendipity while maintaining thematic relevance.

### 8.2. Limitations & Future Improvements
1. **Vocabulary Limitation of TF-IDF**: Bag-of-words / TF-IDF vectors rely on exact lexical or sub-phrase matches. If a user enters synonyms not present in blurbs (e.g., *"thrilling whodunit"* vs. *"detective mystery"*), similarity may be underestimated. Dense semantic embeddings (such as Sentence-BERT) would enhance conceptual matching.
2. **Data Sparsity in Collaborative Filtering**: With 14 readers, the co-rated item intersection between any two readers is small, causing CF to occasionally fall back to global/item means. Matrix Factorization (SVD) or neural collaborative filtering would mitigate this on larger datasets.
3. **Implicit Feedback Integration**: The current system relies on explicit 1–5 ratings. Incorporating implicit interaction signals (clicks, reading time, preview visits) could enrich reader profiles.

---

## 9. Academic Integrity Confirmation

This project was developed independently by **Lương Gia Bảo** (Student ID: `29211358418`, Class: CS467A, Duy Tan University) in fulfillment of Assignment 15 — Individual Mini-Project MP-A02. All algorithms, datasets, tests, and documentation adhere to academic integrity standards.

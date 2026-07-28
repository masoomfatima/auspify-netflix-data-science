# Netflix Content Analysis — Data Science Internship

**Auspify Technologies | Data Science Internship (July–August 2026)**

Live dashboard: **https://netflix-dashboard-masoom.streamlit.app**

Analysis of 8,807 Netflix titles covering data cleaning, exploratory
analysis, machine learning classification, and an interactive dashboard.

---

## Tasks Completed

| # | Task | Notebook |
|---|---|---|
| 1 | Data Cleaning & Preprocessing | `notebooks/01_cleaning.ipynb` |
| 2 | Exploratory Data Analysis | `notebooks/02_eda.ipynb` |
| 5 | Machine Learning Classification | `notebooks/05_classification.ipynb` |
| 6 | Business Insights Dashboard | `dashboard/app.py` |

---

## Key Findings

**Netflix is a movie platform first** — 70% movies, 30% TV shows.

**2019 was peak year** — 1,424 movies added. The decline after that
reflects a strategy shift from licensing external content toward
producing originals.

**India ranks second** (1,008 titles), ahead of the UK. Combined with
"International Movies" being the largest genre (2,752 titles), the
growth story is clearly outside the US.

**TV-MA dominates** (3,207 titles, ~36%) — the target audience is
adults. Kids' content is a visible gap.

---

## Machine Learning: Two Data Leaks

The classification task (predicting Movie vs TV Show) initially scored
98.9% accuracy. That was too good, and investigating revealed why.

**Leak 1 — `duration_unit`:** "min" appears only for movies, "Season"
only for TV shows. A perfect 6,128/0 and 0/2,676 split.

**Leak 2 — genre labels:** Netflix uses separate genre names by type.
"Dramas" is movies-only; "TV Dramas" is TV-only. Only the "Other"
bucket contained both.

### Results after removing both

| Model | Accuracy | TV Show Recall |
|---|---|---|
| Baseline (predict all Movie) | 0.697 | **0.00** |
| Logistic Regression | 0.536 | — |
| Random Forest | **0.672** | **0.70** |

The Random Forest scores below baseline on accuracy but catches 371 of
533 TV shows, while the baseline catches zero. **On imbalanced data,
accuracy is the wrong metric** — recall and F1 tell the real story.

Most important feature: `release_year` (0.394).

**Conclusion:** predicting content type from rating and year alone is
close to impossible. That is itself a finding, not a failure.

---

## Dashboard

Built with Streamlit and Plotly. Features:

- Filters: year range, content type, country
- Live KPIs that update with filters
- Four interactive charts
- A prediction widget running the Random Forest model
- Empty-state handling when filters return no data

Run locally:

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

---

## Structure
├── data/
│ ├── netflix_titles.csv # raw
│ └── netflix_clean.csv # cleaned (8,807 × 18)
├── notebooks/
│ ├── 01_cleaning.ipynb
│ ├── 02_eda.ipynb
│ └── 05_classification.ipynb
├── dashboard/app.py
├── screenshots/ # 8 output images
└── requirements.txt

**Dataset:** [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle, CC0)

**Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit, Plotly
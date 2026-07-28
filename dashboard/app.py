import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Netflix Dashboard", layout="wide")

BASE = Path(__file__).parent.parent


# ---------- Data & Model ----------
@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "netflix_clean.csv")


@st.cache_resource
def train_model(df):
    d = df[['type', 'release_year', 'rating', 'year_added']].dropna()
    y = (d['type'] == 'TV Show').astype(int)
    X = pd.get_dummies(d.drop('type', axis=1), columns=['rating'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X.columns


df = load_data()
model, feature_cols = train_model(df)

st.title("Netflix Content Dashboard")
st.caption("Auspify Technologies — Data Science Internship | Task 6")


# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

years = df['year_added'].dropna()
yr = st.sidebar.slider("Year Added to Netflix",
                       int(years.min()), int(years.max()),
                       (int(years.min()), int(years.max())))

types = st.sidebar.multiselect("Type", df['type'].unique(),
                               default=list(df['type'].unique()))

countries = ['All'] + df['primary_country'].value_counts().head(15).index.tolist()
country = st.sidebar.selectbox("Country", countries)

f = df[(df['year_added'].between(yr[0], yr[1])) & (df['type'].isin(types))]
if country != 'All':
    f = f[f['primary_country'] == country]


# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Titles", len(f))
c2.metric("Movies", (f['type'] == 'Movie').sum())
c3.metric("TV Shows", (f['type'] == 'TV Show').sum())
c4.metric("Countries", f['primary_country'].nunique())

st.divider()


# ---------- Charts ----------
if len(f) == 0:
    st.warning("No data matches these filters. Try adjusting them.")
    st.stop()

left, right = st.columns(2)

with left:
    yearly = f.groupby(['year_added', 'type']).size().reset_index(name='count')
    fig = px.line(yearly, x='year_added', y='count', color='type',
                  title="Content Added Per Year", markers=True,
                  labels={'year_added': 'Year', 'count': 'Titles', 'type': ''},
                  color_discrete_map={'Movie': '#E50914', 'TV Show': '#221F1F'})
    st.plotly_chart(fig, use_container_width=True)

with right:
    top_c = (f[f['primary_country'] != 'Unknown']['primary_country']
             .value_counts().head(10).reset_index())
    top_c.columns = ['country', 'count']
    if len(top_c):
        fig = px.bar(top_c, x='count', y='country', orientation='h',
                     title="Top 10 Countries", color='count',
                     labels={'count': 'Titles', 'country': ''},
                     color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No country data")

left2, right2 = st.columns(2)

with left2:
    genres = (f['listed_in'].str.split(', ').explode()
              .value_counts().head(10).reset_index())
    genres.columns = ['genre', 'count']
    if len(genres):
        fig = px.bar(genres, x='count', y='genre', orientation='h',
                     title="Top 10 Genres", color='count',
                     labels={'count': 'Titles', 'genre': ''},
                     color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No genre data")

with right2:
    ratings = (f[f['rating'] != 'Unknown']['rating']
               .value_counts().head(10).reset_index())
    ratings.columns = ['rating', 'count']
    if len(ratings):
        fig = px.bar(ratings, x='rating', y='count', title="Content Ratings",
                     color='count', labels={'count': 'Titles', 'rating': 'Rating'},
                     color_continuous_scale='Reds')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rating data")

st.divider()


# ---------- Prediction ----------
st.header("Predict: Movie or TV Show?")
st.write("Enter a rating and years to see what the model predicts.")

# The dataset has 3 malformed rows where a duration value ("66 min") ended up
# in the rating column. These are excluded from the dropdown.
valid_ratings = sorted([r for r in df['rating'].dropna().unique()
                        if isinstance(r, str)
                        and not r.endswith('min')
                        and r != 'Unknown'])

p1, p2, p3 = st.columns(3)
in_year = p1.number_input("Release Year", 1940, 2025, 2020)
in_added = p2.number_input("Year Added", 2008, 2025, 2021)
in_rating = p3.selectbox("Rating", valid_ratings)

if st.button("Predict"):
    row = pd.DataFrame([{'release_year': in_year, 'year_added': in_added,
                         'rating': in_rating}])
    row = pd.get_dummies(row, columns=['rating'], drop_first=True)
    row = row.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0]

    if pred == 1:
        st.success(f"**TV Show** — {prob[1]:.0%} confidence")
    else:
        st.info(f"**Movie** — {prob[0]:.0%} confidence")

    st.caption(
        "This model uses only rating and year. Features that directly encode "
        "the answer — duration unit and Netflix's genre labels — were "
        "deliberately excluded. Including them yields 99% accuracy but no real "
        "learning. See the Task 5 notebook for the full leakage analysis."
    )

st.divider()
st.caption("Data: Netflix Movies and TV Shows (Kaggle) | 8,807 titles")
import streamlit as st
import pandas as pd
import numpy as np
import requests, yaml, logging, os
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit_authenticator as stauth

import gspread
from google.oauth2.service_account import Credentials

scope = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
client = gspread.authorize(creds)

SHEET = client.open_by_url("https://docs.google.com/spreadsheets/d/1E1WZ5egjv066mDpwXe4EVOtRLq-gtEEjk1E_dTmW0cs/edit?usp=sharing").sheet1


# ---------- App Config ----------
st.set_page_config(page_title="MovieMate", layout="wide")
logging.basicConfig(level=logging.INFO)

# ---------- Auth: load config ----------
config_path = Path(__file__).parent / "config.yaml"
if not config_path.exists():
    st.error(f"Config not found at {config_path}")
    st.stop()

try:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict) and "credentials" in config and "cookie" in config
except Exception as e:
    st.error(f"Invalid config.yaml: {e}")
    st.stop()

def register_user(username, name, email, password_hash):
    SHEET.append_row([username, name, email, password_hash])


# ---------- Auth: init & login (streamlit-authenticator v0.2.2) ----------
try:
    authenticator = stauth.Authenticate(
        credentials=config["credentials"],
        cookie_name=config["cookie"]["name"],
        key=config["cookie"]["key"],
        cookie_expiry_days=config["cookie"]["expiry_days"],
    )
except Exception as e:
    st.error(f"Auth init error: {e}")
    st.stop()

st.title("🎬 MovieMate — AI Movie Recommender")
if st.toggle("New user? Create an account"):
    username = st.text_input("Choose a Username")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Choose Password", type="password")

    if st.button("Sign Up"):
        if username and password:
            hashed_password = stauth.Hasher([password]).generate()[0]
            register_user(username, name, email, hashed_password)
            st.success("Account created ✅ Now login below.")
        else:
            st.error("Please fill all fields")


name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("❌ Incorrect username or password")
    st.stop()
elif auth_status is None:
    st.warning("⚠️ Please enter your username and password")
    st.stop()

# Authenticated UI
st.sidebar.success(f"✅ Logged in as {name}")
authenticator.logout("Logout", "sidebar")

# ---------- Secrets / Keys ----------
try:
    TMDB_API_KEY = st.secrets["tmdb"]["api_key"]
except Exception:
    st.error("Missing TMDB API key in .streamlit/secrets.toml (tmdb.api_key)")
    st.stop()

# ---------- Cache directory (local, fast) ----------
CACHE_DIR = Path("cachr")
CACHE_DIR.mkdir(exist_ok=True)

# ---------- HTTP session ----------
def create_requests_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    return s

# ---------- TMDB helpers with local cache ----------
def fetch_poster(movie_id: int) -> str:
    cache_file = CACHE_DIR / f"poster_{movie_id}.txt"
    if cache_file.exists():
        return cache_file.read_text()

    session = create_requests_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        r = session.get(url, timeout=(5, 20)); r.raise_for_status()
        data = r.json()
        if data.get("poster_path"):
            poster_url = "https://image.tmdb.org/t/p/w185" + data["poster_path"]
            cache_file.write_text(poster_url)
            return poster_url
    except Exception as e:
        logging.warning(f"Poster fetch failed for {movie_id}: {e}")

    fallback = "https://via.placeholder.com/185x278.png?text=No+Poster"
    cache_file.write_text(fallback)
    return fallback

def fetch_trailer(movie_id: int) -> Optional[str]:
    cache_file = CACHE_DIR / f"trailer_{movie_id}.txt"
    if cache_file.exists():
        saved = cache_file.read_text()
        return saved if saved != "None" else None

    session = create_requests_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
    try:
        r = session.get(url, timeout=(5, 20)); r.raise_for_status()
        results = r.json().get("results", [])

        # Prefer any Trailer on YouTube
        for v in results:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                cache_file.write_text(trailer_url)
                return trailer_url
    except Exception as e:
        logging.warning(f"Trailer fetch failed for {movie_id}: {e}")

    cache_file.write_text("None")
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trending_movies() -> list:
    session = create_requests_session()
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
    out = []
    try:
        r = session.get(url, timeout=(5, 20)); r.raise_for_status()
        for m in r.json().get("results", [])[:5]:
            poster = "https://via.placeholder.com/185x278.png?text=No+Poster"
            if m.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w185" + m["poster_path"]
            trailer = fetch_trailer(m.get("id")) if m.get("id") else None
            out.append({
                "id": m.get("id"),
                "title": m.get("title") or "Untitled",
                "poster": poster,
                "rating": float(m.get("vote_average") or 0.0),
                "year": (m.get("release_date") or "")[:4],
                "trailer": trailer,
            })
    except Exception as e:
        logging.warning(f"Trending fetch failed: {e}")
    return out

# ---------- Theme toggle ----------
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    theme = st.radio("Choose Theme", ["🌞 Light", "🌙 Dark"], key="theme_selector")

# Minimal, clean card UI styles (auto theme)
if theme == "🌙 Dark":
    st.markdown("""
    <style>
    .stApp { background:#0f1115; color:#e6e6e6; }
    .mm-card { padding:1rem; border-radius:14px; background:#151821; box-shadow:0 8px 16px rgba(0,0,0,.25); }
    .mm-btn { background:#1f2430; color:#e6e6e6; border:1px solid #2c3240; border-radius:10px; padding:.5rem 1rem; }
    h1,h2,h3 { color:#fff !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background:#f7f8fb; color:#1d1d1f; }
    .mm-card { padding:1rem; border-radius:14px; background:#ffffff; box-shadow:0 8px 16px rgba(17,17,26,.08); }
    .mm-btn { background:#f1f3f7; color:#1d1d1f; border:1px solid #e6e8ef; border-radius:10px; padding:.5rem 1rem; }
    h1,h2,h3 { color:#111 !important; }
    </style>
    """, unsafe_allow_html=True)

# ---------- Per-user watchlist ----------
WATCHLIST_KEY = f"watchlist_{username}"
if WATCHLIST_KEY not in st.session_state:
    st.session_state[WATCHLIST_KEY] = []
watchlist = st.session_state[WATCHLIST_KEY]

# ---------- Data ----------
DATA_PATH = Path(__file__).parent / "top10K-TMDB-movies.csv"
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.stop()

@st.cache_data(show_spinner=True)
def load_movies(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["genre"] = df["genre"].fillna("").astype(str).str.replace(",", "|")
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["tags"] = df["overview"].fillna("")
    df["title_lc"] = df["title"].fillna("").str.lower()
    return df

movies = load_movies(DATA_PATH)

@st.cache_resource(show_spinner=True)
def build_similarity(df: pd.DataFrame) -> np.ndarray:
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"])
    return cosine_similarity(vectors)

similarity = build_similarity(movies)

def recommend(
    movie: str,
    genres: Optional[List[str]] = None,
    year: Optional[str] = None,
    k: int = 3
) -> Tuple[List[str], List[str], List[str], List[float], List[int], List[Optional[str]], List[int]]:
    title = (movie or "").strip().lower()
    if title not in movies["title_lc"].values:
        return [], [], [], [], [], [], []
    idx = movies.index[movies["title_lc"] == title][0]
    scores = similarity[idx]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:100]

    final_rows = []
    for j, _ in ranked:
        row = movies.iloc[j]
        if genres and not any(g in row["genre"].split("|") for g in genres):
            continue
        if year and year != "Any":
            try:
                if int(row["year"]) != int(year):
                    continue
            except Exception:
                continue
        final_rows.append(row)
        if len(final_rows) >= k:
            break

    if not final_rows:
        return [], [], [], [], [], [], []

    titles = [r.title for r in final_rows]
    ids = [int(r.id) for r in final_rows]
    posters = [fetch_poster(mid) for mid in ids]
    tags = [r.tags for r in final_rows]
    ratings = [float(r.vote_average) if pd.notnull(r.vote_average) else 0.0 for r in final_rows]
    years_ = [int(r.year) if pd.notnull(r.year) else 0 for r in final_rows]
    trailers = [fetch_trailer(mid) for mid in ids]
    return titles, posters, tags, ratings, years_, trailers, ids

# ---------- Sidebar: Filters + Watchlist ----------
all_genres = sorted({g for row in movies["genre"] for g in row.split("|") if g})
selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)
years = sorted(y for y in movies["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Filter by Year", ["Any"] + [str(int(y)) for y in years])

st.sidebar.subheader("🎒 Your Watchlist")
if not watchlist:
    st.sidebar.write("No movies added yet.")
else:
    for mv in watchlist:
        st.sidebar.write(f"🎞 {mv['title']} ({mv['year']}) ⭐ {mv['rating']}")

# ---------- Trending ----------
st.subheader("🔥 Trending This Week")
trending = fetch_trending_movies()
if trending:
    cols = st.columns(len(trending))
    for i, m in enumerate(trending):
        with cols[i]:
            st.markdown(
                f"""
                <div class="mm-card" style="text-align:center">
                    <img src="{m['poster']}" style="width:100%; border-radius:8px;">
                    <h3 style="margin:.6rem 0">{m['title']}</h3>
                    <p style="margin:.2rem 0">📅 {m['year'] or '—'}</p>
                    <p style="margin:.2rem 0">⭐ {m['rating']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("➕ Add to Watchlist", key=f"trend_add_{m['id']}_{i}"):
                item = {"title": m["title"], "year": m["year"], "rating": m["rating"], "poster": m["poster"]}
                if item not in watchlist:
                    watchlist.append(item); st.success("Added ✅")
                else:
                    st.info("Already added 📝")
            if m.get("trailer"):
                st.markdown(f"[🎬 Watch Trailer]({m['trailer']})")

# ---------- Search + Recommendations ----------
movie_name = st.text_input("🔎 Search a movie you liked")
if st.button("Find Recommendations"):
    titles, posters, tags, ratings, years_, trailers, ids = recommend(
        movie_name, selected_genres, selected_year, k=3
    )
    if not titles:
        st.error("No recommendations found. Try another title.")
    else:
        st.subheader("🎬 You may also like")
        cols = st.columns(len(titles))
        for i in range(len(titles)):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="mm-card" style="text-align:center">
                        <img src="{posters[i]}" style="width:100%; border-radius:8px;">
                        <h3 style="margin:.6rem 0">{titles[i]}</h3>
                        <p style="margin:.2rem 0">📅 {years_[i]}</p>
                        <p style="margin:.2rem 0">⭐ {ratings[i]:.1f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("➕ Watchlist", key=f"rec_add_{ids[i]}_{i}"):
                        item = {
                            "title": titles[i],
                            "year": years_[i],
                            "rating": ratings[i],
                            "poster": posters[i],
                            "tags": tags[i],
                        }
                        if item not in watchlist:
                            watchlist.append(item); st.success("Added ✅")
                        else:
                            st.info("Already added 📝")
                with c2:
                    if trailers[i]:
                        st.markdown(f"[🎬 Trailer]({trailers[i]})")

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:.7'>Made with ❤️ by Divyansh_S</p>", unsafe_allow_html=True)
# ---------- End of File ----------
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit_authenticator as stauth
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure page and imports
st.set_page_config(page_title="MovieMate", layout="wide")

# Initialize authentication
try:
    # Load configuration using absolute path
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        st.error(f"Configuration file not found at {config_path}. Please check your setup.")
        st.stop()

    try:
        with config_path.open('r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error reading configuration file: {str(e)}")
        st.stop()

    if not isinstance(config, dict):
        st.error("Invalid configuration format")
        st.stop()

    # Initialize authenticator
    try:
        authenticator = stauth.Authenticate(
            credentials=config['credentials'],
            cookie_name=config['cookie']['name'],
            key=config['cookie']['key'],
            cookie_expiry_days=config['cookie']['expiry_days']
        )
    except Exception as e:
        st.error(f"Error initializing authentication: {str(e)}")
        logging.error(f"Authentication initialization error: {str(e)}")
        st.stop()

    # Display title (after authenticator initialization, before login)
    st.title("🎬 MovieMate – AI Movie Recommender")
    
    # Create the login form in the main area
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    # Handle authentication status
    if authentication_status is False:
        st.error('❌ Username/password is incorrect')
        st.stop()
    elif authentication_status is None:
        st.warning('⚠️ Please enter your username and password')
        st.stop()
    elif authentication_status:
        # Show success message in sidebar
        st.sidebar.success(f"✅ Logged in as {name}")

        # Add logout button to sidebar (use explicit location kwarg)
        # Do not pass unsupported kwargs to logout
        authenticator.logout('Logout', location='sidebar')

        # Initialize user-specific session state
        WATCHLIST_KEY = f"watchlist_{username}"
        if WATCHLIST_KEY not in st.session_state:
            st.session_state[WATCHLIST_KEY] = []
        watchlist = st.session_state[WATCHLIST_KEY]
    else:
        st.error("Something went wrong with authentication")
        st.stop()

except Exception as e:
    logging.error(f"Authentication error: {str(e)}")
    st.error('Authentication system error. Please try again.')
    st.stop()


# ---- API Key ----
TMDB_API_KEY = st.secrets["tmdb"]["api_key"]


def create_requests_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=True
    )
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    return session


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def fetch_trending_movies():
    """Fetch trending movies from TMDB API with enhanced error handling"""
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
    movies_list = []
    
    try:
        session = create_requests_session()
        r = session.get(url, timeout=(5,30))
        r.raise_for_status()
        trending = r.json().get("results", [])[:5]
        
        for m in trending:
            # Get poster with fallback
            poster_url = "https://via.placeholder.com/185x278.png?text=Movie+Poster+Not+Available"
            try:
                if m.get('poster_path'):
                    temp_poster_url = "https://image.tmdb.org/t/p/w185" + m['poster_path']
                    # Verify poster accessibility
                    poster_response = session.get(temp_poster_url, timeout=(3,10))
                    poster_response.raise_for_status()
                    poster_url = temp_poster_url
            except Exception as e:
                logging.warning(f"Could not fetch poster for movie {m.get('title', 'Unknown')}: {str(e)}")
            
            # Get trailer
            trailer_url = None
            try:
                if m.get('id'):
                    trailer_url = fetch_trailer(m['id'])
            except Exception as e:
                logging.warning(f"Could not fetch trailer for movie {m.get('title', 'Unknown')}: {str(e)}")
            
            # Add movie to list
            movies_list.append({
                "title": m.get("title") or "Untitled",
                "poster": poster_url,
                "rating": round(float(m.get("vote_average", 0)), 1),
                "year": (m.get("release_date") or "")[:4],
                "trailer": trailer_url
            })
        return movies_list
    except:
        return []


# ---- Theme Settings ----
# Theme selector in sidebar with better styling
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    theme = st.radio(
        "Choose Theme",
        ["🌞 Light", "🌙 Dark"],
        key="theme_selector"
    )

# Apply theme styles
if theme == "🌙 Dark":
    # Dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton button {
            background-color: #4A4A4A;
            color: #FFFFFF;
            border: 1px solid #666666;
        }
        .stTextInput input {
            background-color: #2D2D2D;
            color: #FFFFFF;
            border: 1px solid #666666;
        }
        .stSelectbox select {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .css-145kmo2 {
            color: #FFFFFF;
        }
        .css-1vq4p4l {
            padding: 1em;
            border-radius: 10px;
            background-color: #2D2D2D;
            margin-bottom: 1em;
        }
        h1, h2, h3 {
            color: #FFFFFF !important;
        }
        .stRadio > div {
            background-color: #2D2D2D;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    # Light theme with better contrast
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stButton button {
            background-color: #F0F2F6;
            color: #000000;
            border: 1px solid #E0E0E0;
        }
        .stTextInput input {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #E0E0E0;
        }
        .stSelectbox select {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stMarkdown {
            color: #000000;
        }
        .css-145kmo2 {
            color: #000000;
        }
        .css-1vq4p4l {
            padding: 1em;
            border-radius: 10px;
            background-color: #F8F9FA;
            margin-bottom: 1em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #000000 !important;
        }
        .stRadio > div {
            background-color: #F8F9FA;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize watchlist
WATCHLIST_KEY = f"watchlist_{username}"
if WATCHLIST_KEY not in st.session_state:
    st.session_state[WATCHLIST_KEY] = []
watchlist = st.session_state[WATCHLIST_KEY]

# ---- Load Movie Dataset ----
movies = pd.read_csv("top10K-TMDB-movies.csv")
movies['genre'] = movies['genre'].fillna('').str.replace(',', '|')
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
movies['tags'] = movies['overview'].fillna('')
movies['title_lc'] = movies['title'].str.lower()


# ---- Sidebar Filters ----
all_genres = sorted({g for row in movies['genre'] for g in row.split('|') if g})
selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)

years = sorted(movies['year'].dropna().unique())
selected_year = st.sidebar.selectbox("Filter by Year", ["Any"] + [str(int(y)) for y in years])


# ---- Watchlist Sidebar ----
st.sidebar.subheader("🎒 Your Watchlist")
if not watchlist:
    st.sidebar.write("No movies added yet.")
else:
    for mv in watchlist:
        st.sidebar.write(f"🎞 {mv['title']} ({mv['year']}) ⭐ {mv['rating']}")


# ---- Similarity Matrix ----
@st.cache_resource
def build_similarity(df: pd.DataFrame) -> np.ndarray:
    """Build similarity matrix from movie tags"""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags'])
    # Convert sparse matrix to dense array directly
    return cosine_similarity(vectors)

# Initialize similarity matrix
similarity = build_similarity(movies)

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """Fetch movie poster from TMDB API"""
    session = create_requests_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        response = session.get(url, timeout=(3, 10))
        response.raise_for_status()
        data = response.json()
        if data.get("poster_path"):
            poster_url = "https://image.tmdb.org/t/p/w185" + data["poster_path"]
            # Verify if the poster image is actually accessible
            img_response = session.get(poster_url, timeout=(3, 10))
            img_response.raise_for_status()
            return poster_url
        logging.warning(f"No poster path found for movie ID {movie_id}")
    except requests.RequestException as e:
        logging.warning(f"Could not fetch poster for movie ID {movie_id}: {str(e)}")
    except Exception as e:
        logging.warning(f"Unexpected error fetching poster for movie ID {movie_id}: {str(e)}")
    return "https://via.placeholder.com/185x278.png?text=Movie+Poster+Not+Available"

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id: int) -> Optional[str]:
    """Fetch movie trailer from TMDB API"""
    session = create_requests_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
    try:
        response = session.get(url, timeout=(3, 10))
        response.raise_for_status()
        data = response.json()
        
        # First try to find official trailer
        for video in data.get("results", []):
            if (video.get("site") == "YouTube" and 
                video.get("type") == "Trailer" and 
                video.get("official", True)):
                return f"https://www.youtube.com/watch?v={video['key']}"
        
        # If no official trailer, try any trailer
        for video in data.get("results", []):
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                return f"https://www.youtube.com/watch?v={video['key']}"
                
        # If no trailer, try any teaser
        for video in data.get("results", []):
            if video.get("site") == "YouTube" and video.get("type") == "Teaser":
                return f"https://www.youtube.com/watch?v={video['key']}"
        
        logging.info(f"No trailer found for movie ID {movie_id}")
    except requests.RequestException as e:
        logging.warning(f"Could not fetch trailer for movie ID {movie_id}: {str(e)}")
    except Exception as e:
        logging.warning(f"Unexpected error fetching trailer for movie ID {movie_id}: {str(e)}")
    return None

def recommend(movie: str, genres: Optional[List[str]] = None, year: Optional[str] = None, k: int = 3) -> Tuple[List[str], List[str], List[str], List[float], List[int], List[Optional[str]]]:
    """Get movie recommendations based on title, genres, and year"""
    title = movie.strip().lower()
    if title not in movies['title_lc'].values:
        return [], [], [], [], [], []

    # Find movie index and get similarity scores
    idx = movies.index[movies['title_lc'] == title][0]
    scores = similarity[idx]
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[1:50]

    final = []
    for j, _ in ranked:
        row = movies.iloc[j]
        # Apply filters
        if genres and not any(g in row['genre'].split('|') for g in genres):
            continue
        if year and year != "Any":
            try:
                if row['year'] != int(year):
                    continue
            except (ValueError, TypeError):
                continue
        final.append(row)
        if len(final) >= k:
            break

    if not final:
        return [], [], [], [], [], []
    
    # Extract required information
    titles = [r.title for r in final]
    posters = [fetch_poster(r.id) for r in final]
    tags = [r.tags for r in final]
    ratings = [float(r.vote_average) if pd.notnull(r.vote_average) else 0.0 for r in final]
    years_ = [int(r.year) if pd.notnull(r.year) else 0 for r in final]
    trailers = [fetch_trailer(r.id) for r in final]
    
    return titles, posters, tags, ratings, years_, trailers

    titles = [r.title for r in final]
    posters = [fetch_poster(r.id) for r in final]
    tags = [r.tags for r in final]
    ratings = [r.vote_average for r in final]
    years_ = [r.year for r in final]
    trailers = [fetch_trailer(r.id) for r in final]
    return titles, posters, tags, ratings, years_, trailers


# ---- Card Styling Based on Theme ----
if theme == "🌙 Dark":
    card_style = """
        padding: 1rem;
        border-radius: 8px;
        background-color: #2D2D2D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    """
    button_style = """
        background-color: #4A4A4A;
        color: white;
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    """
else:
    card_style = """
        padding: 1rem;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    """
    button_style = """
        background-color: #F0F2F6;
        color: black;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    """

# ---- UI ----
st.subheader("🔥 Trending This Week")
trending_movies = fetch_trending_movies()

if trending_movies:
    cols = st.columns(len(trending_movies))
    for i, movie in enumerate(trending_movies):
        with cols[i]:
            # Apply card styling
            st.markdown(f"""
                <div style='{card_style}'>
                    <div style='text-align: center;'>
                        <img src='{movie["poster"]}' style='max-width: 100%; border-radius: 4px;'>
                        <h3 style='margin: 0.5rem 0; font-size: 1.1rem;'>{movie['title']}</h3>
                        <p style='margin: 0.2rem 0;'>📅 {movie['year']}</p>
                        <p style='margin: 0.2rem 0;'>⭐ {movie['rating']:.1f}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add to watchlist button
            if st.button("➕ Add to Watchlist", key=f"trending_add_{movie['title']}"):
                    movie_info = {
                        "title": movie['title'],
                        "year": movie['year'],
                        "rating": movie['rating'],
                        "poster": movie['poster']
                    }
                    if movie_info not in watchlist:
                        watchlist.append(movie_info)
                        st.success("Added to Watchlist ✅")
                    else:
                        st.info("Already in Watchlist 📝")
                
            # If there's a trailer, add a trailer button
            if "trailer" in movie and movie["trailer"]:
                st.markdown(f"[🎬 Watch Trailer]({movie['trailer']})")

movie_name = st.text_input("Search a movie")

if st.button("🔍 Find Recommendations", key="find_recommendations"):
    titles, posters, tags, ratings, years_, trailers = recommend(movie_name, selected_genres, selected_year)
    if not titles:
        st.error("🔍 No recommendations found. Please try another movie title.")
    else:
        st.subheader("🎬 You may also like:")
        
        # Create columns for recommendations
        cols = st.columns(len(titles))
        for i in range(len(titles)):
            with cols[i]:
                # Apply card styling
                st.markdown(f"""
                    <div style='{card_style}'>
                        <div style='text-align: center;'>
                            <img src='{posters[i]}' style='max-width: 100%; border-radius: 4px;'>
                            <h3 style='margin: 0.5rem 0; font-size: 1.1rem;'>{titles[i]}</h3>
                            <p style='margin: 0.2rem 0;'>📅 {years_[i]}</p>
                            <p style='margin: 0.2rem 0;'>⭐ {ratings[i]:.1f}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Buttons container
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("➕ Watchlist", key=f"add_{titles[i]}_{i}"):
                        movie_info = {
                            "title": titles[i],
                            "year": years_[i],
                            "rating": ratings[i],
                            "poster": posters[i],
                            "tags": tags[i]
                        }
                        if movie_info not in watchlist:
                            watchlist.append(movie_info)
                            st.success("Added to Watchlist ✅")
                        else:
                            st.info("Already in Watchlist 📝")
                    
                    # Display trailer button if available
                    if trailers[i]:
                        st.markdown(f"[![🎬]('https://img.shields.io/badge/Watch-Trailer-red')]({trailers[i]})")
                    
                    # Display movie tags/genres in a compact way
                    if tags[i]:
                        genre_list = [tag.strip() for tag in tags[i].split('|')[:3]]
                        st.markdown(f"*{' • '.join(genre_list)}*", help=tags[i])


st.markdown("<footer><p>Made with ❤️ by Divyansh_S</p></footer>", unsafe_allow_html=True)

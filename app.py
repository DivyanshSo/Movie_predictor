import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- API Key ---
TMDB_API_KEY = st.secrets["tmdb"]["api_key"]


#---Trending Movies Section---
@st.cache_data(show_spinner=False)
def fetch_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # get 5 movies only:-
        trending = data.get('results', [])[:5]
        movies_list = []
        for movie in trending:
            poster = (
                "https://image.tmdb.org/t/p/w185" + movie['poster_path']
                if movie.get('poster_path') else "https://via.placeholder.com/185x278.png?text=No+Image"
            )
            movies_list.append({
                'title': movie['title'],
                'poster': poster,
                'rating': movie.get('vote_average', 'N/A'),  # <-- fix here
                'year': movie.get('release_date', '')[:4],
            })
        return movies_list
    except Exception as e:
        st.warning(f"Error fetching trending movies: {e}")
        return []
    
#---UI---page
st.set_page_config(page_title="MovieMate", layout="wide")
st.title("🎬 MovieMate – AI Movie Recommender")

# Show Trending Movies
st.subheader("🔥 Trending This Week")
trending_movies = fetch_trending_movies()
if trending_movies:
    cols = st.columns(len(trending_movies))
    for i, movie in enumerate(trending_movies):
        with cols[i]:
            st.image(movie["poster"], use_container_width=True)
            st.caption(f"{movie['title']} ({movie['year']})")
            st.write(f"⭐ {movie['rating']}")
else:
    st.info("Trending movies not available at the moment.")

# --- Theme Toggle ---
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #222; color: #eee; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #fff; color: #000; }
        .css-1d391kg { color: #222 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load data
movies = pd.read_csv("top10K-TMDB-movies.csv")

# Preprocess genres and year
movies['genre'] = movies['genre'].fillna('').apply(lambda x: '|'.join([g['name'] for g in eval(x)]) if x.startswith('[') else x)
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
movies['tags'] = movies['overview'].fillna('')

# --- Genre and Year Filters ---
all_genres = sorted(set(g for genres in movies['genre'].dropna() for g in genres.split('|') if g))
selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)
years = sorted(movies['year'].dropna().unique())
selected_year = st.sidebar.selectbox("Filter by Year", ["Any"] + [str(int(y)) for y in years])

# --- Cached Similarity ---
@st.cache_resource
def get_similarity_and_vectors(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = get_similarity_and_vectors(movies)


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w185" + poster_path
    except Exception:
        pass
    return "https://via.placeholder.com/185x278.png?text=No+Image"

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        for video in data.get('results', []):
            if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                return f"https://www.youtube.com/watch?v={video['key']}"
    except Exception:
        pass
    return None

user_reviews = {}

def recommend(movie, genres=None, year=None):
    movie = movie.lower()
    filtered_movies = movies.copy()
    if genres:
        filtered_movies = filtered_movies[filtered_movies['genre'].apply(lambda x: any(g in x.split('|') for g in genres))]
    if year and year != "Any":
        filtered_movies = filtered_movies[filtered_movies['year'] == int(year)]
    if movie not in filtered_movies['title'].str.lower().values:
        return [], [], [], [], [], []
    
    idx = filtered_movies[filtered_movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]  # top 3

    titles, posters, tags, ratings, years_, trailers = [], [], [], [], [], []
    for i in movie_list:
        row = filtered_movies.iloc[i[0]]
        titles.append(row.title)
        posters.append(fetch_poster(row.id))
        tags.append(row.tags)
        ratings.append(row.vote_average)
        years_.append(row.year)
        trailers.append(fetch_trailer(row.id))
    
    return titles, posters, tags, ratings, years_, trailers

# --- UI ---
st.set_page_config(page_title="MovieMate", layout="wide")
st.title("🎬 MovieMate – AI Movie Recommender")

movie_name = st.text_input("Search your favorite movie", "")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        with st.spinner("Fetching recommendations..."):
            titles, posters, tags, ratings, years_, trailers = recommend(movie_name, selected_genres, selected_year)
        valid = min(len(titles), len(posters))
        if valid == 0:
            st.error("Movie not found or no recommendations available.")
        else:
            st.subheader("You may also like:")
            cols = st.columns(valid)
            for i in range(valid):
                with cols[i]:
                    st.image(posters[i], use_container_width=True)
                    st.caption(f"🎞 {titles[i]} ({years_[i]})")
                    st.write(f"🧾 {tags[i]}")
                    st.write(f"⭐ Rating: {ratings[i]}")
                    if trailers[i]:
                        st.markdown(f"[▶ Watch Trailer]({trailers[i]})")

                    review_key = f"{titles[i]}_{years_[i]}"
                    with st.expander(f"Write a review for {titles[i]}"):
                        user_review = st.text_area("Your Review:", key=review_key)
                        if st.button(f"Submit Review", key=f"btn_{review_key}"):
                            user_reviews[review_key] = user_review
                            st.success("Review submitted!")
                        if review_key in user_reviews:
                            st.info(f"Your review: {user_reviews[review_key]}")

# --- Footer ---
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp {bottom: 0;}
    </style>
    <footer>
        <p>Made with ❤️ by Divyansh_S</p>
    </footer>
""", unsafe_allow_html=True)

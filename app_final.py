import streamlit as st 
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Movie Dashboard Recommender App", page_icon=":bar_chart:", layout="wide")

def generate_wordcloud(text, colormap='YlOrBr'):
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap=colormap).generate(text)
    return wordcloud

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Movies_IMDb.xlsx")
        
        df['Genres'] = df['Genres'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
        df['Genres'] = df['Genres'].apply(lambda x: list(set(x)))  # Remove duplicates in each movie's genre list
        df['Stars'] = df['Stars'].apply(lambda x: x.split(",") if isinstance(x, str) else [])
        genres = df['Genres'].str.get_dummies(sep=', ')
        df = pd.concat([df, genres], axis=1)
        
        df['Duration_In_Minutes'] = pd.to_numeric(df['Duration_In_Minutes'].str.replace('min', ''), errors='coerce')
        df['Stars'] = df['Stars'].apply(lambda x: [s.strip() for s in x])
        all_stars = df['Stars'].explode().str.strip()  
        unique_stars = all_stars.nunique()
            
        return df, genres, unique_stars, all_stars
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0

df, genres, unique_stars, all_stars = load_data()

if 'Director' in df.columns:
    unique_directors = df['Director'].nunique()
else:
    unique_directors = 0

# Set custom CSS for the theme
st.markdown("""
    <style>
        body {
            background-color: #000; /* Black background */
            color: #FFD700; /* Gold text */
        }
        .main .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-top: 20px;
        }
        .sidebar .block-container {
            background-color: #1a1a1a; /* Darker sidebar */
            color: #FFD700;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Arial', sans-serif;
            color: #FFD700; /* Gold headers */
        }
        .stMetric {
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 5px;
            margin: 10px;
        }
        .dashed-line {
            border-top: 2px dashed #FFD700; /* Gold dashed line */
            margin: 20px 0;
            width: 100%;
        }
        .stTextInput label, .stTextArea label, .stSelectbox label {
            color: #FFD700; /* Gold text for labels */
            font-size: 18px; /* Larger font size for labels */
        }
    </style>
""", unsafe_allow_html=True)

# Pages in sidebar
pages = ["Overview", "Deep Insights", "Movie Recommendation System"] 
page = st.sidebar.radio("📂 Pages", pages)

# Filters for all pages
if page in ["Overview", "Deep Insights"]:
    st.sidebar.title("🔍 Filters")
    # Filters for Year, MPAA, and Genres
    year_filter = st.sidebar.multiselect(
        "Select Year",
        options=["All"] + sorted(df["Year"].dropna().astype(int).unique().tolist()),
        default=["All"]
    )
    mpaa_filter = st.sidebar.multiselect(
        "Select MPAA Rating",
        options=["All"] + sorted(df["MPAA"].dropna().unique().tolist()),
        default=["All"]
    )
    genre_filter = st.sidebar.multiselect(
        "Select Genres",
        options=["All"] + sorted(set(g for sublist in df['Genres'] for g in sublist)),
        default=["All"]
    )

    # Apply filters
    filtered_df = df.copy()
    if "All" not in year_filter:
        filtered_df = filtered_df[filtered_df["Year"].isin(year_filter)]
    if "All" not in mpaa_filter:
        filtered_df = filtered_df[filtered_df["MPAA"].isin(mpaa_filter)]

    # Ensure 'Genres' column is properly handled when filtering
    if "All" not in genre_filter:
        filtered_df = filtered_df[filtered_df['Genres'].apply(lambda genres: any(genre in genre_filter for genre in genres))]

    # Get unique genres from the 'Genres' column (after splitting and deduplicating)
    unique_genres = sorted(set(g for sublist in df['Genres'] for g in sublist))


if page == "Overview":
    st.markdown('<h1 style="text-align: center;">🎬 IMDb Movies Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)

    # Key Metrics
    st.markdown("<h2 style='text-align: center;'>📊 Key Metrics</h2>", unsafe_allow_html=True)
    # Custom CSS to style metrics with borders
    st.markdown("""
        <style>
            .metric-container {
                text-align: center;
                margin-bottom: 20px;
                padding: 10px; /* Padding for spacing inside each metric */
                border: 1px solid #FFD700; /* Gold border for separation */
                border-radius: 8px; /* Rounded corners for a modern look */
                background-color: #2C2F33; /* Dark background to enhance readability */
            }
            .metric-label {
                font-size: 24px;  /* Larger font size for titles */
                color: #FFFFFF;  /* White color for titles */
                font-weight: bold;
            }
            .metric-value {
                font-size: 32px;  /* Bigger font size for values */
                font-weight: bold;
                color: #FFFFFF;  /* White color for values */
            }
        </style>
    """, unsafe_allow_html=True)

    # Metrics layout
    col1, col2, col3, col4 = st.columns(4)

    # Total Movies
    with col1:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Movies</div>
                <div class="metric-value">{len(filtered_df)}</div>
            </div>
        """, unsafe_allow_html=True)

    # Unique Genres
    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Unique Genres</div>
                <div class="metric-value">{len(unique_genres)}</div>
            </div>
        """, unsafe_allow_html=True)

    # Unique MPAA Ratings
    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Unique MPAA Ratings</div>
                <div class="metric-value">{filtered_df['MPAA'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)

    # Unique Directors
    with col4:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Unique Directors</div>
                <div class="metric-value">{unique_directors}</div>
            </div>
        """, unsafe_allow_html=True)

    # Additional metrics
    col5, col6, col7 = st.columns(3)

    # Unique Stars
    with col5:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Unique Stars</div>
                <div class="metric-value">{unique_stars}</div>
            </div>
        """, unsafe_allow_html=True)

    # Average Duration
    with col6:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Average Duration</div>
                <div class="metric-value">{filtered_df['Duration_In_Minutes'].mean():.2f} mins</div>
            </div>
        """, unsafe_allow_html=True)

    # Average IMDb Rating
    with col7:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Average IMDb Rating</div>
                <div class="metric-value">{filtered_df['IMDb_Rating'].mean():.1f}</div>
            </div>
        """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Movies Released Each Year")
        movies_by_year = filtered_df.groupby('Year').size()
        
        fig_movies_by_year = px.line(
            movies_by_year, 
            x=movies_by_year.index, 
            y=movies_by_year.values
            )
        
        st.plotly_chart(fig_movies_by_year, use_container_width=True)

    with col2:
        st.subheader("📈 Average IMDb Ratings of Movies Over the Years")
        movies_by_imdb_rating = filtered_df.groupby('Year')['IMDb_Rating'].mean()
        
        fig_movies_by_imdb_rating = px.line(
            movies_by_imdb_rating, 
            x=movies_by_imdb_rating.index, 
            y=movies_by_imdb_rating.values, 
            color_discrete_sequence=["orange"]
        )
        st.plotly_chart(fig_movies_by_imdb_rating, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⭐ Top 5 Stars by Number of Movies")
        all_stars = all_stars[all_stars != "Unknown"]
        top_5_stars = all_stars.value_counts().head(5)
        fig_top_5_stars = px.bar(
            x=top_5_stars.index,
            y=top_5_stars.values,
            labels={'x': 'Star', 'y': 'Number of Movies'},
            color_discrete_sequence=['#BFAEE3']
        )
        st.plotly_chart(fig_top_5_stars, use_container_width=True)
        
    with col2:
        st.subheader("🎞️ Top 5 Genres by Number of Movies")
        top_5_genres = df['Genres'].explode().value_counts().head(5)
        fig_top_5_genres = px.bar(
            x=top_5_genres.index,
            y=top_5_genres.values,
            labels={'x': 'Genre', 'y': 'Number of Movies'},
            color_discrete_sequence=['#FEC5E6']
        )
        st.plotly_chart(fig_top_5_genres, use_container_width=True)

elif page == "Deep Insights":
    st.markdown('<h1 style="text-align: center;">🔍 Deep Insights</h1>', unsafe_allow_html=True)
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌟 Top 5 MPAA Ratings by Number of Movies")
        top_5_mpaa = filtered_df['MPAA'].value_counts().head(5).sort_values(ascending=False)
        fig_top_5_mpaa = px.bar(
            x=top_5_mpaa.index,
            y=top_5_mpaa.values,
            labels={'x': 'MPAA Rating', 'y': 'Number of Movies'},
            color_discrete_sequence=['#4EB09B']
        )
        st.plotly_chart(fig_top_5_mpaa, use_container_width=True)
    with col2:
        st.subheader("⏱️ Average Duration of Movies by MPAA Rating")
        avg_duration_by_mpaa = filtered_df.groupby('MPAA')['Duration_In_Minutes'].mean().sort_values(ascending=False)
        fig_avg_duration_by_mpaa = px.bar(
            avg_duration_by_mpaa,
            x=avg_duration_by_mpaa.index,
            y=avg_duration_by_mpaa.values,
            labels={'x': 'MPAA Rating', 'y': 'Average Duration (minutes)'},
            color_discrete_sequence=['#A8CD89']
        )
        st.plotly_chart(fig_avg_duration_by_mpaa, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 Genres with Highest Average IMDb Ratings")
        avg_imdb_by_genre = df.explode('Genres').groupby('Genres')['IMDb_Rating'].mean().sort_values(ascending=False).head(10)
        fig_avg_imdb_by_genre = px.bar(
            avg_imdb_by_genre,
            x=avg_imdb_by_genre.values,
            y=avg_imdb_by_genre.index,
            orientation='h',
            labels={'x': 'Average IMDb Rating', 'y': 'Genre'}
        )
        st.plotly_chart(fig_avg_imdb_by_genre, use_container_width=True)
    
    with col2:
        st.subheader("📅 Number of Movies by Duration Category")
        short_movies = filtered_df[filtered_df['Duration_In_Minutes'] < 90]
        middle_movies = filtered_df[(filtered_df['Duration_In_Minutes'] >= 90) & (filtered_df['Duration_In_Minutes'] <= 120)]
        long_movies = filtered_df[filtered_df['Duration_In_Minutes'] > 120]

        movie_categories = pd.DataFrame({
            "Category": ["Short Movies (<90 mins)", "Middle Movies (90-120 mins)", "Long Movies (>120 mins)"],
            "Number of Movies": [short_movies.shape[0], middle_movies.shape[0], long_movies.shape[0]]
        })

        fig_duration_category_counts = px.pie(
            movie_categories,
            names='Category',
            values='Number of Movies',
            color_discrete_sequence=['#C7D7FB', '#F1B0DA', '#FEC4B6'],
            hole=0.6
        )
        
        st.plotly_chart(fig_duration_category_counts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Popular Directors")
        directors_text = " ".join(filtered_df.explode('Director')['Director'].dropna())
        directors_wordcloud = generate_wordcloud(directors_text)
        plt.imshow(directors_wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    with col2:
        st.subheader("📖 Popular Plot Summary")
        plot_summary_text = " ".join(filtered_df['Plot_Summary'].dropna())
        plot_summary_wordcloud = generate_wordcloud(plot_summary_text)
        plt.imshow(plot_summary_wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        
elif page == "Movie Recommendation System":
    
    st.markdown('<h1 style="text-align: center;">🎥 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.write('<div style="text-align: center; font-size: 15px;">This system recommends movies based on multiple search conditions such as movie title, genres, star, and plot summary.</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
    
    df['Title'] = df['Title'].fillna('')
    df['Genres'] = df['Genres'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['Stars'] = df['Stars'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['Director'] = df['Director'].fillna('')
    df['Plot_Summary'] = df['Plot_Summary'].fillna('')

    df['combined_features'] = (
        df['Title'] + ' ' +
        df['Genres'] + ' ' +
        df['Stars'] + ' ' +
        df['Director'] + ' ' +
        df['Plot_Summary']
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    def multi_condition_recommendation(query_dict, movies_data, tfidf_vectorizer, tfidf_matrix, top_n=5):
        valid_keys = ['Title', 'Genres', 'Stars', 'Director', 'Plot_Summary']
        total_similarity = np.zeros(tfidf_matrix.shape[0])
        num_conditions = 0

        # Calculate similarity for each condition
        for query_type, query_input in query_dict.items():
            query_type_normalized = query_type.strip().title().replace(" ", "_")
            if query_type_normalized not in valid_keys:
                continue

            # Generate query vector using TF-IDF
            query_vector_tfidf = tfidf_vectorizer.transform([query_input])
            cosine_sim = cosine_similarity(query_vector_tfidf, tfidf_matrix).flatten()
            total_similarity += cosine_sim
            num_conditions += 1

        if num_conditions > 0:
            total_similarity /= num_conditions

        # Get top recommendations
        similar_indices = np.argsort(total_similarity)[-top_n - 1:-1][::-1]
        recommendations = [(movies_data.iloc[i]['Title'], total_similarity[i]) for i in similar_indices]
        return recommendations


    title_input = st.selectbox("Movie title to search for:", options=[""] + df['Title'].tolist())
    genres = st.text_input("Genre to search for:")
    stars = st.text_input("Star to search for:")
    director = st.text_input("Director to search for:")
    plot_summary = st.text_area("Plot summary to search for:")

    query_dict = {}
    if title_input:
        query_dict['Title'] = title_input
    if genres:
        query_dict['Genres'] = genres
    if stars:
        query_dict['Stars'] = stars
    if director:
        query_dict['Director'] = director
    if plot_summary:
        query_dict['Plot_Summary'] = plot_summary

    if query_dict:
        with st.spinner('Searching...'):
            recommendations = multi_condition_recommendation(query_dict, df, tfidf_vectorizer, tfidf_matrix, top_n=5)

            st.subheader("Recommended movies for you:")
            for idx, (title, score) in enumerate(recommendations):
                st.write(f"{idx + 1}. {title}")
    else:
        st.write("Please enter at least one search condition to get recommendation.")

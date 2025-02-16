# IMDB Movies Dashboard Recommender App
An interactive **Streamlit** application for visualizing and analyzing IMDb movie data, along with a Movies Recommendation System built using TF-IDF & Cosine Similarity-based NLP techniques. This dashboard provides insights into various movie statistics, including the total number of movies, genres, IMDb ratings, and more. It also features interactive filters for customized analysis and personalized movie recommendations.

## 👋 Features  

### **Overview Page**  
- General statistics and charts displaying the number of movies released per year, IMDb rating trends, and top contributors in genres and stars.  

### **Deep Insights Page**  
- Detailed analysis of MPAA ratings, movie durations, IMDb ratings by genre, and notable directors and movie titles.  

### **Interactive Filters**  
- Customize the dashboard by filtering movies based on **year, MPAA rating, and genre**.  

### **Movies Recommendation System**  
- Build a TF-IDF & Cosine Similarity-based movie recommender using NLP techniques.  

## 🚀 Running the Application  

Start the **Streamlit** app:  
```bash
streamlit run app_final.py
```
Access the dashboard:  
Open your browser and go to **http://localhost:8501**  

Deploy on **Streamlit Community Cloud**: **[Click here](https://imdb-movies-dashboard-recommender.streamlit.app/)**

## 📝 Requirements  

Ensure the following dependencies are installed:  

- **Python 3.10+**  
- **pip** (Python package manager)  

Install required libraries:  
```bash
pip install streamlit pandas plotly wordcloud matplotlib numpy scikit-learn openpyxl
```



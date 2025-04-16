# Movie-Recommendation-System
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample movie dataset
data = {
    'title': ['The Dark Knight', 'Inception', 'Interstellar', 'The Shawshank Redemption', 'Pulp Fiction'],
    'genre': ['Action, Crime, Drama', 'Action, Adventure, Sci-Fi', 'Adventure, Drama, Sci-Fi', 'Drama', 'Crime, Drama'],
    'description': ['When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                    'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                    'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                    'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                    'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.'
                   ]
}

# Create DataFrame
movies_df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'genre' column
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genre'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies_df['title'].iloc[movie_indices]

# Example usage
recommended_movies = recommend_movies('Inception')
print(recommended_movies)

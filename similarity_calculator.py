import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    @staticmethod
    def calculate_cosine_similarity(embedding1, embedding2):
        try:
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)
            similarity_score = cosine_similarity(embedding1, embedding2)
            return similarity_score[0][0]
        except ValueError as e:
            print(f"Error in calculating cosine similarity: {e}")
            return 0  # Return 0 or an appropriate fallback value
        except Exception as e:
            print(f"Unexpected error in similarity calculation: {e}")
            return 0

    @staticmethod
    def get_word_embedding(value, embeddings_cache, nlp_model):
        try:
            if isinstance(value, list):
                value = ' '.join(map(str, value))

            if value not in embeddings_cache:
                embeddings_cache[value] = nlp_model(str(value)).vector
            return embeddings_cache[value]
        except Exception as e:
            print(f"Error generating word embedding for value '{value}': {e}")
            return np.zeros(nlp_model.vocab.vectors_length)  # Return a zero vector as fallback


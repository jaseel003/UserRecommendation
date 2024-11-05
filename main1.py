# import time
# import spacy
# from dotenv import load_dotenv
# import os
# from data_processor import DataProcessor
# from user_sim import UserSimilarityAnalyzer
# from key_comparator import find_comparable_keys_by_module, get_top_comparable_keys_by_module
# import json

# def main():
#     # Load environment variables from .env file
#     load_dotenv()

#     start_time = time.time()

#     # MongoDB connection details from environment variables
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("DB_NAME")
#     collection_name = os.getenv("COLLECTION_NAME", "new_sample_collection")
#     output_filename = os.getenv("OUTPUT_FILENAME", "outputmatch1.txt")  # Changed filename
#     key_comparison_output = os.getenv("KEY_COMPARISON_OUTPUT", "key_comparison_results.json")  # New output file for key comparisons
#     threshold = float(os.getenv("THRESHOLD", 0.6))  # Default to 0.6 if not set
#     sample_size = int(os.getenv("SAMPLE_SIZE", 10))  # Changed sample size to 10

#     # Initialize the DataProcessor with the connection details
#     data_processor = DataProcessor(mongo_uri, db_name, collection_name)
#     user_similarity_analyzer = UserSimilarityAnalyzer()

#     try:
#         # Fetch data from MongoDB
#         data = data_processor.fetch_data()
#         nlp_model = spacy.load("en_core_web_md")
        
#         # Generate all key-value pairs for similarity analysis, limited to sample_size per role
#         sampled_key_value_pairs = user_similarity_analyzer.generate_key_value_pairs(data, sample_size=sample_size)
        
#         # Initialize an empty embeddings cache
#         embeddings_cache = {}
        
#         # Calculate similarity scores
#         similarity_data = user_similarity_analyzer.calculate_similarity_scores(sampled_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold)
        
#         # Perform key comparison
#         comparable_keys_by_module = find_comparable_keys_by_module(similarity_data, threshold)
#         top_comparable_keys = get_top_comparable_keys_by_module(comparable_keys_by_module, top_n=5)

#         # Store the top comparable keys to a file
#         with open(key_comparison_output, "w") as f:  # Save to key_comparison_output
#             json.dump(top_comparable_keys, f, indent=4)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # Ensure the MongoDB connection is closed
#         data_processor.close_connection()
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Total Execution Time: {execution_time} seconds")

# if __name__ == "__main__":
#     main()



# fulldata
import time
import spacy
from dotenv import load_dotenv
import os
from data_processor import DataProcessor
from user_similarity_analyzer import UserSimilarityAnalyzer
from user2 import UserSimilarityAnalyzerFull
from key_comparator import find_comparable_keys_by_module, get_top_comparable_keys_by_module

def main():
    # Load environment variables from .env file
    load_dotenv()

    start_time = time.time()

    # MongoDB connection details from environment variables
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLLECTION_NAME", "new_sample_collection")
    output_filename_sample = os.getenv("OUTPUT_FILENAME", "outputmatch1.txt")  # For sampled similarity results
    output_filename_full = os.getenv("FULL_SIMILARITY_OUTPUT", "similarity.txt")  # For full similarity results
    threshold = float(os.getenv("THRESHOLD", 0.6))  # Default threshold of 0.6
    sample_size = int(os.getenv("SAMPLE_SIZE", 10))  # Sample size for each role

    # Initialize the DataProcessor with the connection details
    data_processor = DataProcessor(mongo_uri, db_name, collection_name)
    user_similarity_analyzer = UserSimilarityAnalyzer()
    user_similarity_analyzer_full = UserSimilarityAnalyzerFull()

    try:
        # Fetch data from MongoDB
        data = data_processor.fetch_data()
        nlp_model = spacy.load("en_core_web_md")
        
        ### Step 1: Sample Data and Perform Key Comparison ###

        # Generate key-value pairs limited to the sample size for similarity analysis
        sampled_key_value_pairs = user_similarity_analyzer.generate_key_value_pairs(data, sample_size=sample_size)
        
        # Initialize an empty embeddings cache
        embeddings_cache = {}
        
        # Calculate similarity scores for the sampled data
        similarity_data_sample = user_similarity_analyzer.calculate_similarity_scores(
            sampled_key_value_pairs, embeddings_cache, nlp_model, output_filename_sample, threshold
        )
        
        # Perform key comparison
        comparable_keys_by_module = find_comparable_keys_by_module(similarity_data_sample, threshold)
        top_comparable_keys = get_top_comparable_keys_by_module(comparable_keys_by_module, top_n=5)

        ### Step 2: Load Allowed Keys and Perform Full Similarity Analysis ###

        # Load allowed keys into the UserSimilarityAnalyzerFull
        user_similarity_analyzer_full.initialize_allowed_keys(top_comparable_keys)

        # Generate all key-value pairs for full similarity analysis
        all_key_value_pairs = user_similarity_analyzer_full.generate_key_value_pairs_full(data)

        # Calculate similarity scores for the full data using the allowed keys
        user_similarity_analyzer_full.calculate_similarity_scores_full(
            all_key_value_pairs, embeddings_cache, nlp_model, output_filename_full, threshold
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the MongoDB connection is closed
        data_processor.close_connection()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total Execution Time: {execution_time} seconds")

if __name__ == "__main__":
    main()



# random sample selection
# import random  # Import random for random sampling
# from dask import delayed, compute
# from similarity_calculator import SimilarityCalculator
# from file_writer import FileWriter

# class UserSimilarityAnalyzer:
#     @staticmethod
#     def generate_key_value_pairs(data, sample_size=5):
#         """
#         Generate key-value pairs from the nested dictionary structure, limited to a random sample of size per role.
#         """
#         key_value_pairs = []
#         try:
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict):
#                         key_value_pairs.extend(UserSimilarityAnalyzer.generate_key_value_pairs(item, sample_size))
#             elif isinstance(data, dict):
#                 for module, roles_data in data.items():
#                     if isinstance(roles_data, dict):
#                         for role, role_data in roles_data.items():
#                             # Select a random sample from role_data, ensuring sample size doesn't exceed available data
#                             limited_role_data = random.sample(role_data, min(sample_size, len(role_data)))
#                             role_index = len(key_value_pairs) + 1
#                             for user_index, item in enumerate(limited_role_data, start=1):
#                                 temp_item = {k: v for k, v in item.items() if k != 'id'}
#                                 key_value_pairs.append((module, role, role_index, user_index, temp_item))
#         except Exception as e:
#             print(f"Error generating key-value pairs: {e}")
#         return key_value_pairs

#     @staticmethod
#     def _calculate_similarity_for_pair(pair1, all_key_value_pairs, embeddings_cache, nlp_model, threshold):
#         """
#         Calculate similarity scores for a pair of key-value pairs.
#         """
#         results = []
#         module1, role1, role1_index, user1_index, user1 = pair1

#         for (module2, role2, role2_index, user2_index, user2) in all_key_value_pairs:
            
#             if role1 == role2:
#                 continue  # Skip if the roles are the same (if that rule is needed)

#             for key1, value1 in user1.items():
#                 for key2, value2 in user2.items():
#                     embedding1 = SimilarityCalculator.get_word_embedding(value1, embeddings_cache, nlp_model)
#                     embedding2 = SimilarityCalculator.get_word_embedding(value2, embeddings_cache, nlp_model)
#                     similarity_score = SimilarityCalculator.calculate_cosine_similarity(embedding1, embedding2)

#                     # Ensure compatibility with JSON serialization
#                     similarity_score = float(similarity_score)

#                     if similarity_score >= threshold:
#                         results.append({
#                             "user1": {
#                                 "module": module1,
#                                 "role": role1,
#                                 "role_index": role1_index,
#                                 "user_index": user1_index,
#                                 "key": key1,
#                                 "value": value1
#                             },
#                             "user2": {
#                                 "module": module2,
#                                 "role": role2,
#                                 "role_index": role2_index,
#                                 "user_index": user2_index,
#                                 "key": key2,
#                                 "value": value2
#                             },
#                             "similarity_score": similarity_score
#                         })
#         return results

#     @staticmethod
#     def calculate_similarity_scores(all_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold):
#         """
#         Calculate similarity scores for all key-value pairs using Dask and write results to file.
#         """
#         results = []
#         selected_similarity_count = 0  # Counter for selected similarity scores
#         similarity_tasks = []

#         try:
#             # Create Dask tasks for each key-value pair comparison
#             for pair in all_key_value_pairs:
#                 task = delayed(UserSimilarityAnalyzer._calculate_similarity_for_pair)(
#                     pair, all_key_value_pairs, embeddings_cache, nlp_model, threshold
#                 )
#                 similarity_tasks.append(task)

#             # Compute all tasks in parallel
#             task_results = compute(*similarity_tasks, scheduler='threads')

#             # Aggregate results from all tasks
#             for result in task_results:
#                 results.extend(result)
#                 selected_similarity_count += len(result)

#             # Write results to JSON file
#             FileWriter.write_similarity_scores(output_filename, results)
#             FileWriter.write_similarity_count(output_filename, selected_similarity_count)

#         except Exception as e:
#             print(f"Error calculating similarity scores: {e}")
        
#         return results  # Return the results for key comparison



# fisrt5

# from dask import delayed, compute
# from similarity_calculator import SimilarityCalculator
# from file_writer import FileWriter

# class UserSimilarityAnalyzer:
#     @staticmethod
#     def generate_key_value_pairs(data, sample_size=5):
#         """
#         Generate key-value pairs from the nested dictionary structure, limited to a sample size per role.
#         """
#         key_value_pairs = []
#         try:
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict):
#                         key_value_pairs.extend(UserSimilarityAnalyzer.generate_key_value_pairs(item, sample_size))
#             elif isinstance(data, dict):
#                 for module, roles_data in data.items():
#                     if isinstance(roles_data, dict):
#                         for role, role_data in roles_data.items():
#                             # Limit role_data to the first `sample_size` entries
#                             limited_role_data = role_data[:sample_size]
#                             role_index = len(key_value_pairs) + 1
#                             for user_index, item in enumerate(limited_role_data, start=1):
#                                 temp_item = {k: v for k, v in item.items() if k != 'id'}
#                                 key_value_pairs.append((module, role, role_index, user_index, temp_item))
#         except Exception as e:
#             print(f"Error generating key-value pairs: {e}")
#         return key_value_pairs

#     @staticmethod
#     def _calculate_similarity_for_pair(pair1, all_key_value_pairs, embeddings_cache, nlp_model, threshold):
#         """
#         Calculate similarity scores for a pair of key-value pairs.
#         """
#         results = []
#         module1, role1, role1_index, user1_index, user1 = pair1

#         for (module2, role2, role2_index, user2_index, user2) in all_key_value_pairs:
            
#             if role1 == role2:
#                 continue  # Skip if the roles are the same (if that rule is needed)

#             for key1, value1 in user1.items():
#                 for key2, value2 in user2.items():
#                     embedding1 = SimilarityCalculator.get_word_embedding(value1, embeddings_cache, nlp_model)
#                     embedding2 = SimilarityCalculator.get_word_embedding(value2, embeddings_cache, nlp_model)
#                     similarity_score = SimilarityCalculator.calculate_cosine_similarity(embedding1, embedding2)

#                     # Ensure compatibility with JSON serialization
#                     similarity_score = float(similarity_score)

#                     if similarity_score >= threshold:
#                         results.append({
#                             "user1": {
#                                 "module": module1,
#                                 "role": role1,
#                                 "role_index": role1_index,
#                                 "user_index": user1_index,
#                                 "key": key1,
#                                 "value": value1
#                             },
#                             "user2": {
#                                 "module": module2,
#                                 "role": role2,
#                                 "role_index": role2_index,
#                                 "user_index": user2_index,
#                                 "key": key2,
#                                 "value": value2
#                             },
#                             "similarity_score": similarity_score
#                         })
#         return results

#     @staticmethod
#     def calculate_similarity_scores(all_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold):
#         """
#         Calculate similarity scores for all key-value pairs using Dask and write results to file.
#         """
#         results = []
#         selected_similarity_count = 0  # Counter for selected similarity scores
#         similarity_tasks = []

#         try:
#             # Create Dask tasks for each key-value pair comparison
#             for pair in all_key_value_pairs:
#                 task = delayed(UserSimilarityAnalyzer._calculate_similarity_for_pair)(
#                     pair, all_key_value_pairs, embeddings_cache, nlp_model, threshold
#                 )
#                 similarity_tasks.append(task)

#             # Compute all tasks in parallel
#             task_results = compute(*similarity_tasks, scheduler='threads')

#             # Aggregate results from all tasks
#             for result in task_results:
#                 results.extend(result)
#                 selected_similarity_count += len(result)

#             # Write results to JSON file
#             FileWriter.write_similarity_scores(output_filename, results)
#             FileWriter.write_similarity_count(output_filename, selected_similarity_count)

#         except Exception as e:
#             print(f"Error calculating similarity scores: {e}")
        
#         return results  # Return the results for key comparison




 
# exception 
import random  # Import random for random sampling
from dask import delayed, compute
from similarity_calculator import SimilarityCalculator
from file_writer import FileWriter
from collections import defaultdict

class UserSimilarityAnalyzer:
    EXCLUDED_KEYS = {"name", "phone", "contact", "mobile number"}

    @staticmethod
    def generate_key_value_pairs(data, sample_size=5):
        """
        Generate key-value pairs from the nested dictionary structure, limited to a random sample of size per role.
        Excludes specific keys defined in EXCLUDED_KEYS.
        """
        key_value_pairs = []
        try:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        key_value_pairs.extend(UserSimilarityAnalyzer.generate_key_value_pairs(item, sample_size))
            elif isinstance(data, dict):
                for module, roles_data in data.items():
                    if isinstance(roles_data, dict):
                        for role, role_data in roles_data.items():
                            # Select a random sample from role_data, ensuring sample size doesn't exceed available data
                            limited_role_data = random.sample(role_data, min(sample_size, len(role_data)))
                            role_index = len(key_value_pairs) + 1
                            for user_index, item in enumerate(limited_role_data, start=1):
                                # Exclude specific keys
                                temp_item = {k: v for k, v in item.items() if k != 'id'}
                                key_value_pairs.append((module, role, role_index, user_index, temp_item))
        except Exception as e:
            print(f"Error generating key-value pairs: {e}")
        return key_value_pairs

    @staticmethod
    def _calculate_similarity_for_pair(pair1, all_key_value_pairs, embeddings_cache, nlp_model, threshold):
        """
        Calculate similarity scores for a pair of key-value pairs, excluding specified keys from consideration.
        """
        results = []
        module1, role1, role1_index, user1_index, user1 = pair1

        for (module2, role2, role2_index, user2_index, user2) in all_key_value_pairs:
            if role1 == role2:
                continue  # Skip if the roles are the same (if that rule is needed)

            for key1, value1 in user1.items():
                # Skip excluded keys during similarity calculation
                if key1.lower() in UserSimilarityAnalyzer.EXCLUDED_KEYS:
                    continue
                
                for key2, value2 in user2.items():
                    # Skip excluded keys during similarity calculation
                    if key2.lower() in UserSimilarityAnalyzer.EXCLUDED_KEYS:
                        continue

                    embedding1 = SimilarityCalculator.get_word_embedding(value1, embeddings_cache, nlp_model)
                    embedding2 = SimilarityCalculator.get_word_embedding(value2, embeddings_cache, nlp_model)
                    similarity_score = SimilarityCalculator.calculate_cosine_similarity(embedding1, embedding2)

                    # Ensure compatibility with JSON serialization
                    similarity_score = float(similarity_score)

                    if similarity_score >= threshold:
                        results.append({
                            "user1": {
                                "module": module1,
                                "role": role1,
                                "role_index": role1_index,
                                "user_index": user1_index,
                                "key": key1,
                                "value": value1
                            },
                            "user2": {
                                "module": module2,
                                "role": role2,
                                "role_index": role2_index,
                                "user_index": user2_index,
                                "key": key2,
                                "value": value2
                            },
                            "similarity_score": similarity_score
                        })
        return results

    @staticmethod
    def calculate_similarity_scores(all_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold):
        """
        Calculate similarity scores for all key-value pairs using Dask and write results to file.
        """
        results = []
        selected_similarity_count = 0  # Counter for selected similarity scores
        similarity_tasks = []

        try:
            # Create Dask tasks for each key-value pair comparison
            for pair in all_key_value_pairs:
                task = delayed(UserSimilarityAnalyzer._calculate_similarity_for_pair)(
                    pair, all_key_value_pairs, embeddings_cache, nlp_model, threshold
                )
                similarity_tasks.append(task)

            # Compute all tasks in parallel
            task_results = compute(*similarity_tasks, scheduler='threads')

            # Aggregate results from all tasks
            for result in task_results:
                results.extend(result)
                selected_similarity_count += len(result)

            # Write results to JSON file
            FileWriter.write_similarity_scores(output_filename, results)
            FileWriter.write_similarity_count(output_filename, selected_similarity_count)

        except Exception as e:
            print(f"Error calculating similarity scores: {e}")
        
        return results  # Return the results for key comparison

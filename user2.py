import json
from dask import delayed, compute
from similarity_calculator import SimilarityCalculator
from file_writer import FileWriter

class UserSimilarityAnalyzerFull:
    allowed_keys = {}

    @staticmethod
    def initialize_allowed_keys(allowed_keys):
        """Set allowed keys from the given dictionary."""
        UserSimilarityAnalyzerFull.allowed_keys = allowed_keys

    @staticmethod
    def _filter_keys(user_data, module, role):
        """Filter user data to keep only allowed keys."""
        allowed = UserSimilarityAnalyzerFull.allowed_keys.get(module, {}).get(role, [])
        return {key: value for key, value in user_data.items() if key in allowed}

    @staticmethod
    def generate_key_value_pairs_full(data):
        """Generate key-value pairs from the nested dictionary structure."""
        key_value_pairs = []
        try:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        key_value_pairs.extend(UserSimilarityAnalyzerFull.generate_key_value_pairs_full(item))
            elif isinstance(data, dict):
                for module, roles_data in data.items():
                    if isinstance(roles_data, dict):
                        for role, role_data in roles_data.items():
                            if isinstance(role_data, list):
                                for user_index, item in enumerate(role_data, start=1):
                                    temp_item = UserSimilarityAnalyzerFull._filter_keys(item, module, role)
                                    key_value_pairs.append((module, role, None, user_index, temp_item))
        except Exception as e:
            print(f"Error generating key-value pairs: {e}")
        return key_value_pairs

    @staticmethod
    def _calculate_similarity_for_pair_full(pair1, all_key_value_pairs, embeddings_cache, nlp_model, threshold):
        """Calculate similarity scores for a pair of key-value pairs."""
        results = []
        module1, role1, _, user1_index, user1 = pair1

        # Filter user1 data to only include allowed keys
        user1_filtered = UserSimilarityAnalyzerFull._filter_keys(user1, module1, role1)

        for (module2, role2, _, user2_index, user2) in all_key_value_pairs:
            # Filter user2 data to only include allowed keys
            user2_filtered = UserSimilarityAnalyzerFull._filter_keys(user2, module2, role2)

            if role1 == role2 or not user1_filtered or not user2_filtered:
                continue  # Skip if the roles are the same or if filtered data is empty

            for key1, value1 in user1_filtered.items():
                for key2, value2 in user2_filtered.items():
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
                                "user_index": user1_index,
                                "key": key1,
                                "value": value1
                            },
                            "user2": {
                                "module": module2,
                                "role": role2,
                                "user_index": user2_index,
                                "key": key2,
                                "value": value2
                            },
                            "similarity_score": similarity_score
                        })
        return results

    @staticmethod
    def calculate_similarity_scores_full(all_key_value_pairs, embeddings_cache, nlp_model, output_filename, threshold):
        """Calculate similarity scores for all key-value pairs using Dask and write results to file."""
        results = []
        selected_similarity_count = 0  # Counter for selected similarity scores
        similarity_tasks = []

        try:
            # Create Dask tasks for each key-value pair comparison
            for pair in all_key_value_pairs:
                task = delayed(UserSimilarityAnalyzerFull._calculate_similarity_for_pair_full)(
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
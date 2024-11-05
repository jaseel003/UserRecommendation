# import json
# from collections import defaultdict

# def load_similarity_data(filename):
#     """
#     Load the similarity results from the output file.
#     """
#     try:
#         with open(filename, "r") as file:
#             return json.load(file)
#     except Exception as e:
#         print(f"Error loading file {filename}: {e}")
#         return []

# def find_comparable_keys_by_module(similarity_data, similarity_threshold=0.6):
#     """
#     Identify the comparable keys for each module with high similarity scores.
#     """
#     comparable_keys_by_module = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # module -> role -> key -> frequency

#     # Iterate through the similarity results
#     for result in similarity_data:
#         # Ensure the result contains user1 and user2
#         user1 = result.get('user1')
#         user2 = result.get('user2')
#         similarity_score = result.get('similarity_score', 0)

#         if user1 and user2:
#             user1_module = user1.get('module')
#             user1_role = user1.get('role')  # Extract the role for user1
#             user1_key = user1.get('key')
#             user2_module = user2.get('module')
#             user2_role = user2.get('role')  # Extract the role for user2
#             user2_key = user2.get('key')

#             # Only count if similarity score meets the threshold
#             if similarity_score >= similarity_threshold:
#                 if user1_key and user1_module and user1_role:
#                     comparable_keys_by_module[user1_module][user1_role][user1_key] += 1
#                 if user2_key and user2_module and user2_role:
#                     comparable_keys_by_module[user2_module][user2_role][user2_key] += 1
#         else:
#             print(f"Skipping invalid result: {result}")

#     return {module: {role: dict(keys) for role, keys in roles.items()} for module, roles in comparable_keys_by_module.items()}

# def get_top_comparable_keys_by_module(comparable_keys_by_module, top_n=5):
#     """
#     Get the top N comparable keys for each module and role.
#     """
#     top_comparable_keys_by_module = {}

#     for module, roles in comparable_keys_by_module.items():
#         top_comparable_keys_by_module[module] = {}
#         for role, keys in roles.items():
#             sorted_comparable_keys = sorted(keys.items(), key=lambda x: x[1], reverse=True)
#             top_comparable_keys_by_module[module][role] = [key for key, count in sorted_comparable_keys[:top_n]]

#     return top_comparable_keys_by_module
import json
from collections import defaultdict

def load_similarity_data(filename):
    """
    Load the similarity results from the output file.
    """
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return []

def find_comparable_keys_by_module(similarity_data, similarity_threshold=0.6):
    """
    Identify the comparable keys for each module with high similarity scores.
    """
    comparable_keys_by_module = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # module -> role -> key -> frequency

    # Iterate through the similarity results
    for result in similarity_data:
        user1 = result.get('user1')
        user2 = result.get('user2')
        similarity_score = result.get('similarity_score', 0)

        if user1 and user2:
            user1_module = user1.get('module')
            user1_role = user1.get('role')
            user1_key = user1.get('key')
            user2_module = user2.get('module')
            user2_role = user2.get('role')
            user2_key = user2.get('key')

            # Only count if similarity score meets the threshold
            if similarity_score >= similarity_threshold:
                if user1_key and user1_module and user1_role:
                    comparable_keys_by_module[user1_module][user1_role][user1_key] += 1
                if user2_key and user2_module and user2_role:
                    comparable_keys_by_module[user2_module][user2_role][user2_key] += 1
        else:
            print(f"Skipping invalid result: {result}")

    return {module: {role: dict(keys) for role, keys in roles.items()} for module, roles in comparable_keys_by_module.items()}

def get_top_comparable_keys_by_module(comparable_keys_by_module, top_n=5):
    """
    Get the top N comparable keys for each module and role.
    """
    top_comparable_keys_by_module = {}

    for module, roles in comparable_keys_by_module.items():
        top_comparable_keys_by_module[module] = {}
        for role, keys in roles.items():
            sorted_comparable_keys = sorted(keys.items(), key=lambda x: x[1], reverse=True)
            top_comparable_keys_by_module[module][role] = [key for key, count in sorted_comparable_keys[:top_n]]

    return top_comparable_keys_by_module


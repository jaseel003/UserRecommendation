import json

class FileWriter:
    @staticmethod
    def write_similarity_scores(output_filename, results):
        try:
            with open(output_filename, "w") as file:
                json.dump(results, file, indent=4)
        except Exception as e:
            print(f"Error writing similarity scores to file {output_filename}: {e}")

    @staticmethod
    def write_similarity_count(output_filename, selected_similarity_count):
        try:
            # Read the existing JSON file
            with open(output_filename, "r") as file:
                data = json.load(file)

            # Add the similarity count
            data.append({"selected_similarity_count": selected_similarity_count})

            # Write the updated JSON back to the file
            with open(output_filename, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error writing similarity count to file {output_filename}: {e}")




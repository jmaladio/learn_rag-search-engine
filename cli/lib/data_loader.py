import json

def load_movies(file_path: str) -> list[dict]:
    """
    Loads movies from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing movie data.
    Returns:
        list[dict]: A list of dictionaries, each representing a movie.

    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return data["movies"] if isinstance(data, dict) else data
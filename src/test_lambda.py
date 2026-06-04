from lambda_function import lambda_handler
import json

event = {
    "body": json.dumps({
        "danceability": 0.7,
        "energy": 0.8,
        "loudness": -5,
        "valence": 0.6,
        "speechiness": 0.05,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "acousticness": 0.2,
        "duration_ms": 210000,
        "key": 5,
        "mode": 1,
        "explicit": 0,
        "track_name": "Test song",
        "artists": "Artist A",
        "album_name": "Test album",
        "track_genre": "pop"
    })
}

response = lambda_handler(event, None)
print(response)

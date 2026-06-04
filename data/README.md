# Data

The dataset used in this project was provided exclusively for an academic ML competition at Universidad de los Andes and is **not publicly available**.

It is therefore not included in this repository. To replicate the results, you would need access to an equivalent Spotify tracks dataset containing the following variables:

## Variables Used

| Variable | Type | Description |
|---|---|---|
| `popularity` | int | Target variable (0–100) |
| `track_name` | string | Song title |
| `artists` | string | Artist name(s) |
| `album_name` | string | Album name |
| `track_genre` | string | Music genre |
| `danceability` | float | Danceability score (0–1) |
| `energy` | float | Energy score (0–1) |
| `loudness` | float | Loudness in dB |
| `valence` | float | Musical positivity (0–1) |
| `speechiness` | float | Speech presence (0–1) |
| `instrumentalness` | float | Instrumentalness score (0–1) |
| `liveness` | float | Live performance likelihood (0–1) |
| `acousticness` | float | Acousticness score (0–1) |
| `duration_ms` | int | Track duration in milliseconds |
| `key` | int | Musical key |
| `mode` | int | Modality (major=1, minor=0) |
| `explicit` | bool | Explicit content flag |

## Alternative Dataset

A similar publicly available dataset can be found on Kaggle:
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

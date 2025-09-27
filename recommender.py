
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os

# --------------------------
# Config
# --------------------------
RATINGS_PATH = '/content/data/ratings.csv'
MOVIES_PATH = '/content/data/movies.csv'
K = 10             # Precision@K / Recall@K
THRESHOLD = 3.5    # rating threshold to consider an item "relevant"
MF_STEPS = 50
MF_ALPHA = 0.0002
MF_BETA = 0.02

# --------------------------
# Load data
# --------------------------
ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

# Pivot to user-item matrix (rows = userId, cols = movieId)
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_ids = user_item_matrix.index.values           # array of userId (may not be 0..N-1)
movie_ids = user_item_matrix.columns.values        # array of movieId

num_users, num_items = user_item_matrix.shape
num_factors = 10

# --------------------------
# Initialize latent matrices
# --------------------------
rng = np.random.default_rng(seed=42)
user_matrix = rng.random((num_users, num_factors))
item_matrix = rng.random((num_items, num_factors))

# --------------------------
# Matrix factorization (basic SGD as in your original)
# --------------------------
def matrix_factorization(R, P, Q, steps=MF_STEPS, alpha=MF_ALPHA, beta=MF_BETA):
    Q = Q.copy()
    P = P.copy()
    for step in range(steps):
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[j, :].T)
                    P[i, :] += alpha * (2 * eij * Q[j, :] - beta * P[i, :])
                    Q[j, :] += alpha * (2 * eij * P[i, :] - beta * Q[j, :])
        # optional early stopping by error
        error = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j] > 0:
                    error += (R[i][j] - np.dot(P[i, :], Q[j, :].T)) ** 2
                    error += (beta / 2) * (np.sum(P[i, :] ** 2) + np.sum(Q[j, :] ** 2))
        if error < 0.001:
            break
    return P, Q

user_matrix, item_matrix = matrix_factorization(user_item_matrix.values, user_matrix, item_matrix)

# --------------------------
# Predicted matrix (users x items)
# --------------------------
predicted_matrix = np.dot(user_matrix, item_matrix.T)

# --------------------------
# Robust recommend function (maps to movieId & title)
# --------------------------
# mapping movieId -> column index
movieid_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

def recommend_movies_for_user(user_id, top_n=10):
    # find user index in the pivot table
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
    except KeyError:
        raise ValueError(f"user_id {user_id} not found in data")

    preds = predicted_matrix[user_idx, :]
    top_idxs = np.argsort(preds)[-top_n:][::-1]  # indices into movie_ids
    top_movie_ids = movie_ids[top_idxs]

    top_df = pd.DataFrame({
        'movieId': top_movie_ids,
        'predicted_rating': preds[top_idxs]
    })
    # merge movie titles
    top_df = top_df.merge(movies, on='movieId', how='left')
    # order by predicted_rating desc
    return top_df[['movieId', 'title', 'predicted_rating']]

# Example
example_user = user_ids[0] if len(user_ids) > 0 else None
if example_user is not None:
    print(f"Top recommendations for user {example_user}:")
    print(recommend_movies_for_user(example_user, top_n=5).head())

# --------------------------
# Evaluation: overall RMSE using only rated entries
# --------------------------
mask_rated = user_item_matrix.values > 0
actual_rated = user_item_matrix.values[mask_rated]
predicted_rated = predicted_matrix[mask_rated]
overall_rmse = np.sqrt(mean_squared_error(actual_rated, predicted_rated))
print(f"Overall RMSE (on rated entries): {overall_rmse:.4f}")

# --------------------------
# Per-user RMSE and Precision@K / Recall@K
# --------------------------
per_user_rows = []
precisions = []
recalls = []

for u_idx, u_id in enumerate(user_ids):
    actual = user_item_matrix.values[u_idx, :]
    predicted = predicted_matrix[u_idx, :]

    rated_mask = actual > 0
    num_rated = int(rated_mask.sum())

    if num_rated > 0:
        # user RMSE on rated items
        user_rmse = np.sqrt(mean_squared_error(actual[rated_mask], predicted[rated_mask]))

        # relevant items for this user (actual >= threshold)
        relevant = set(np.where(actual >= THRESHOLD)[0])

        # top-K predicted indices
        top_k = list(np.argsort(predicted)[-K:][::-1])
        hits = len(set(top_k) & relevant)

        precision_u = hits / K
        recall_u = hits / len(relevant) if len(relevant) > 0 else np.nan
    else:
        user_rmse = np.nan
        precision_u = np.nan
        recall_u = np.nan

    per_user_rows.append({
        'userId': int(u_id),
        'num_ratings': num_rated,
        'rmse': user_rmse,
        f'precision@{K}': precision_u,
        f'recall@{K}': recall_u
    })

    if not np.isnan(precision_u):
        precisions.append(precision_u)
    if not np.isnan(recall_u):
        recalls.append(recall_u)

per_user_df = pd.DataFrame(per_user_rows)
per_user_df.to_csv('per_user_metrics.csv', index=False)
print("Saved per_user_metrics.csv")

# --------------------------
# Per-movie metrics (RMSE, counts, avg actual/predicted)
# --------------------------
per_movie_rows = []
for m_idx, m_id in enumerate(movie_ids):
    actual = user_item_matrix.values[:, m_idx]
    predicted = predicted_matrix[:, m_idx]
    rated_mask = actual > 0
    num_rated = int(rated_mask.sum())

    if num_rated > 0:
        movie_rmse = np.sqrt(mean_squared_error(actual[rated_mask], predicted[rated_mask]))
        avg_actual = actual[rated_mask].mean()
        avg_predicted = predicted[rated_mask].mean()
    else:
        movie_rmse = np.nan
        avg_actual = np.nan
        avg_predicted = np.nan

    per_movie_rows.append({
        'movieId': int(m_id),
        'num_ratings': num_rated,
        'rmse': movie_rmse,
        'avg_actual_rating': avg_actual,
        'avg_predicted_rating': avg_predicted
    })

per_movie_df = pd.DataFrame(per_movie_rows)
# attach titles
per_movie_df = per_movie_df.merge(movies[['movieId','title']], on='movieId', how='left')
per_movie_df.to_csv('per_movie_metrics.csv', index=False)
print("Saved per_movie_metrics.csv")

# --------------------------
# Overall Precision@K & Recall@K (mean over users)
# --------------------------
mean_precision_at_k = np.nanmean(per_user_df[f'precision@{K}'].values)
mean_recall_at_k = np.nanmean(per_user_df[f'recall@{K}'].values)
print(f"Precision@{K} (mean over users): {mean_precision_at_k:.4f}")
print(f"Recall@{K} (mean over users): {mean_recall_at_k:.4f}")

# --------------------------
# Export predictions for Tableau
# - all pairs (may be large)
# - rated-only (smaller)
# We'll also include movie titles for convenience
# --------------------------
# Flatten all pairs
all_user_ids = np.repeat(user_ids, num_items)
all_movie_ids = np.tile(movie_ids, num_users)
all_actuals = user_item_matrix.values.flatten()
all_predicted = predicted_matrix.flatten()

preds_all_df = pd.DataFrame({
    'userId': all_user_ids.astype(int),
    'movieId': all_movie_ids.astype(int),
    'actual_rating': all_actuals,
    'predicted_rating': all_predicted
})
# merge movie titles
preds_all_df = preds_all_df.merge(movies[['movieId','title']], on='movieId', how='left')
preds_all_df.to_csv('recommender_predictions_all.csv', index=False)
print("Saved recommender_predictions_all.csv (all user-item pairs)")

# Rated-only
preds_rated_df = preds_all_df[preds_all_df['actual_rating'] > 0].copy()
preds_rated_df.to_csv('recommender_predictions_rated.csv', index=False)
print("Saved recommender_predictions_rated.csv (only rated pairs)")

# --------------------------
# Save summary metrics and append to a metrics history (timestamped)
# --------------------------
metrics_summary = {
    'timestamp': datetime.utcnow().isoformat(),
    'overall_rmse': overall_rmse,
    f'precision@{K}': mean_precision_at_k,
    f'recall@{K}': mean_recall_at_k,
    'num_users': num_users,
    'num_items': num_items
}
metrics_df = pd.DataFrame([metrics_summary])
metrics_df.to_csv('recommender_metrics.csv', index=False)
print("Saved recommender_metrics.csv (current run summary)")

# append to history
history_path = 'recommender_metrics_history.csv'
if os.path.exists(history_path):
    hist = pd.read_csv(history_path)
    hist = pd.concat([hist, metrics_df], ignore_index=True)
else:
    hist = metrics_df
hist.to_csv(history_path, index=False)
print(f"Appended run to {history_path}")

print("✅ Exports ready for Tableau: recommender_predictions_all.csv, recommender_predictions_rated.csv, per_user_metrics.csv, per_movie_metrics.csv, recommender_metrics.csv, recommender_metrics_history.csv")

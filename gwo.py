import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# batas sesuai judul: n_estimators, max_depth, min_samples_split, min_samples_leaf
BOUNDS = np.array([
    [100, 800],   # n_estimators
    [3, 30],     # max_depth
    [2, 20],     # min_samples_split
    [1, 10],     # min_samples_leaf
], dtype=float)


def decode_position(position):
    p = np.rint(position).astype(int)
    return {
        "n_estimators": int(p[0]),
        "max_depth": int(p[1]),
        "min_samples_split": int(p[2]),
        "min_samples_leaf": int(p[3]),
    }


def fitness(position, X_train, y_train, random_state=42):
    params = decode_position(position)
    model = RandomForestClassifier(
        **params,
        random_state=random_state,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)    
    return scores.mean()


def grey_wolf_optimizer(X_train, y_train, n_wolves=8, max_iter=12, random_state=42):
    rng = np.random.default_rng(random_state)
    dim = BOUNDS.shape[0]
    wolves = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_wolves, dim))

    alpha_pos = beta_pos = delta_pos = None
    alpha_score = beta_score = delta_score = -np.inf
    history = []

    for iteration in range(max_iter):
        for i in range(n_wolves):
            wolves[i] = np.clip(wolves[i], BOUNDS[:, 0], BOUNDS[:, 1])
            score = fitness(wolves[i], X_train, y_train, random_state)

            if score > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = score, wolves[i].copy()
            elif score > beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = score, wolves[i].copy()
            elif score > delta_score:
                delta_score, delta_pos = score, wolves[i].copy()

        a = 2 - iteration * (2 / max_iter)
        for i in range(n_wolves):
            new_position = np.zeros(dim)
            for leader_pos in [alpha_pos, beta_pos, delta_pos]:
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader_pos - wolves[i])
                X_leader = leader_pos - A * D
                new_position += X_leader / 3

            wolves[i] = np.clip(new_position, BOUNDS[:, 0], BOUNDS[:, 1])

        history.append({
            "iteration": iteration + 1,
            "best_accuracy_cv": float(alpha_score),
            **decode_position(alpha_pos)
        })

        print(f"Iterasi {iteration+1}/{max_iter} | best CV Accuracy: {alpha_score:.4f} | {decode_position(alpha_pos)}")

    return decode_position(alpha_pos), float(alpha_score), history
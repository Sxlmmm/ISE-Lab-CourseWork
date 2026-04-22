import os
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# datasets and their sensitive columns
# I used the same ones from the lab 
DATASETS = {
    "adult": {
        "path": "dataset/processed_adult.csv",
        "target": "Class-label",
        "sensitive": ["race", "gender"],
    },
    "compas": {
        "path": "dataset/processed_compas.csv",
        "target": "Recidivism",
        "sensitive": ["Race", "Sex"],
    },
    "german": {
        "path": "dataset/processed_german.csv",
        "target": "CREDITRATING",
        "sensitive": ["PersonStatusSex", "AgeInYears"],
    },
    "credit": {
        "path": "dataset/processed_credit_with_numerical.csv",
        "target": "class",
        "sensitive": ["SEX", "AGE"],
    },
    "law_school": {
        "path": "dataset/processed_law_school.csv",
        "target": "pass_bar",
        "sensitive": ["race", "male"],
    },
    "communities": {
        "path": "dataset/processed_communities_crime.csv",
        "target": "class",
        "sensitive": ["Black"],
    },
    "dutch": {
        "path": "dataset/processed_dutch.csv",
        "target": "occupation",
        "sensitive": ["sex"],
    },
    "kdd": {
        "path": "dataset/processed_kdd.csv",
        "target": "income",
        "sensitive": ["race", "sex"],
    },
}


# load and split the data
def load_data(path, target):
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c != target]
    X = np.array(df[feature_cols], dtype=np.float32)
    y = np.array(df[target], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = np.array(scaler.fit_transform(X_train), dtype=np.float32)
    X_test  = np.array(scaler.transform(X_test), dtype=np.float32)

    # keep a copy of raw test values for the sensitive column flipping later
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, feature_cols, X_test_raw


# build and train the DNN
# simple architecture - 64 -> 32 -> 1 with dropout to avoid overfitting
def train_model(X_train, y_train, X_test, y_test, name):
    model = keras.Model(
        *( lambda inp: (inp, layers.Dense(1, activation="sigmoid")(
            layers.Dense(32, activation="relu")(
            layers.Dropout(0.2)(
            layers.Dense(64, activation="relu")(inp))))))(
            layers.Input(shape=(X_train.shape[1],)))
    )

    # rewrite above more clearly
    inp = layers.Input(shape=(X_train.shape[1],))
    x   = layers.Dense(64, activation="relu")(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=25, batch_size=512,
              validation_data=(X_test, y_test), verbose=0)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  [{name}] model accuracy: {acc:.4f}")
    return model



# helper functions used by both methods
def get_sensitive_idx(feature_cols, sensitive_cols):
    return [i for i, c in enumerate(feature_cols) if c in sensitive_cols]

def get_nonsensitive_idx(feature_cols, sensitive_cols):
    return [i for i, c in enumerate(feature_cols) if c not in sensitive_cols]

def get_col_stds(X_raw, ns_idx):
    # standard deviation per column, used to scale perturbations
    return np.array([X_raw[:, i].std() + 1e-8 for i in ns_idx], dtype=np.float32)

def flip_sensitive(population, s_idx, X_raw):
    # for each sensitive column, randomly pick a different observed value
    counterfactuals = population.copy()
    for col_idx in s_idx:
        unique_vals = np.unique(X_raw[:, col_idx])
        for row in range(len(counterfactuals)):
            current = counterfactuals[row, col_idx]
            other_vals = unique_vals[unique_vals != current]
            if len(other_vals) > 0:
                counterfactuals[row, col_idx] = np.random.choice(other_vals)
    return counterfactuals

def predict(model, X):
    return model(X, training=False).numpy().ravel()

def get_discrimination_scores(model, population, s_idx, X_raw, threshold=0.05):
    # score = |P(x) - P(x')| where x' is the counterfactual
    cf = flip_sensitive(population, s_idx, X_raw)
    scores = np.abs(predict(model, population) - predict(model, cf)).astype(np.float32)
    is_discriminatory = scores > threshold
    return scores, is_discriminatory



# BASELINE: Random Search (from lab 4)
# just randomly sample test inputs and check if they are discriminatory

def random_search(model, X_test, X_raw, feature_cols, sensitive_cols,
                  n_samples=500, threshold=0.05):

    s_idx  = get_sensitive_idx(feature_cols, sensitive_cols)
    ns_idx = get_nonsensitive_idx(feature_cols, sensitive_cols)
    stds   = get_col_stds(X_raw, ns_idx)

    # randomly pick samples and add a small perturbation to non-sensitive features
    indices = np.random.randint(len(X_test), size=n_samples)
    A = X_test[indices].copy()
    for j, i in enumerate(ns_idx):
        A[:, i] += np.random.normal(0, 0.1 * stds[j], size=n_samples).astype(np.float32)

    # create counterfactuals and add same perturbation
    B = flip_sensitive(A, s_idx, X_raw)
    for j, i in enumerate(ns_idx):
        B[:, i] += np.random.normal(0, 0.1 * stds[j], size=n_samples).astype(np.float32)

    pa = predict(model, A)
    pb = predict(model, B)
    idi = float(np.mean(np.abs(pa - pb) > threshold))
    return idi


# GA helper functions

def tournament_selection(population, fitness_scores, k=3):
    # pick k random individuals and return the best one
    candidates = np.random.choice(len(population), size=k, replace=False)
    winner = candidates[np.argmax(fitness_scores[candidates])]
    return population[winner].copy()

def crossover(parent1, parent2, ns_idx):
    # uniform crossover - each non-sensitive gene comes from either parent randomly
    child = parent1.copy()
    for i in ns_idx:
        if np.random.rand() < 0.5:
            child[i] = parent2[i]
    return child

def mutate(individual, ns_idx, stds, mutation_prob=0.1, mutation_scale=0.05):
    # gaussian mutation on non-sensitive features
    mutated = individual.copy()
    for j, i in enumerate(ns_idx):
        if np.random.rand() < mutation_prob:
            mutated[i] += np.float32(np.random.normal(0, mutation_scale * stds[j]))
    return mutated


# -----------------------------------------------------------------------
# PROPOSED METHOD: Genetic Algorithm
#
# The idea is to evolve a population of test inputs towards regions
# where the model is most discriminatory. Instead of sampling randomly
# like the baseline, the GA uses selection, crossover and mutation to
# guide the search. This is inspired by search-based testing approaches
# like Aequitas but uses a full evolutionary strategy.
#
# GA settings (tried a few combos, these worked best):
#   pop_size = 50  (100 was slower with similar results)
#   elite_size = 5 (keep top 5 each gen so we don't lose good solutions)
#   crossover_prob = 0.8
#   mutation_prob = 0.1 per gene
#   tournament_k = 3
# -----------------------------------------------------------------------
def genetic_algorithm(model, X_test, X_raw, feature_cols, sensitive_cols,
                      n_samples=500, threshold=0.05,
                      pop_size=50, elite_size=5,
                      crossover_prob=0.8, mutation_prob=0.1,
                      mutation_scale=0.05, tournament_k=3):

    s_idx  = get_sensitive_idx(feature_cols, sensitive_cols)
    ns_idx = get_nonsensitive_idx(feature_cols, sensitive_cols)
    stds   = get_col_stds(X_raw, ns_idx)

    # work out how many generations we can afford within the budget
    n_generations = max(1, (n_samples - pop_size) // pop_size)
    total_evals   = pop_size + n_generations * pop_size  # should be close to n_samples
    disc_count    = 0

    # initialise population from random test samples
    init_idx = np.random.randint(len(X_test), size=pop_size)
    population = X_test[init_idx].copy()

    fitness, is_disc = get_discrimination_scores(model, population, s_idx, X_raw, threshold)
    disc_count += int(is_disc.sum())

    # evolve for n_generations
    for gen in range(n_generations):
        next_gen = []

        # elitism - carry over the best individuals unchanged
        elite_idx = np.argsort(fitness)[-elite_size:]
        for idx in elite_idx:
            next_gen.append(population[idx].copy())

        # fill the rest of the population with offspring
        while len(next_gen) < pop_size:
            p1 = tournament_selection(population, fitness, k=tournament_k)
            p2 = tournament_selection(population, fitness, k=tournament_k)

            if np.random.rand() < crossover_prob:
                child = crossover(p1, p2, ns_idx)
            else:
                child = p1.copy()

            child = mutate(child, ns_idx, stds, mutation_prob, mutation_scale)
            next_gen.append(child)

        population = np.array(next_gen, dtype=np.float32)

        fitness, is_disc = get_discrimination_scores(model, population, s_idx, X_raw, threshold)
        disc_count += int(is_disc.sum())

    idi = disc_count / total_evals
    return idi


# run experiment for one dataset
def run_experiment(ds_name, config, n_samples=500, n_trials=10):
    print(f"\n{'='*55}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*55}")

    X_train, X_test, y_train, y_test, feature_cols, X_raw = load_data(
        config["path"], config["target"]
    )

    # only keep sensitive cols that actually exist in the dataset
    sensitive_cols = [c for c in config["sensitive"] if c in feature_cols]
    if not sensitive_cols:
        print(f"  WARNING: no sensitive columns found, skipping")
        return None

    model = train_model(X_train, y_train, X_test, y_test, ds_name)

    rs_scores = []
    ga_scores = []

    for trial in range(n_trials):
        np.random.seed(trial * 17 + 3)

        rs = random_search(model, X_test, X_raw, feature_cols, sensitive_cols, n_samples)
        ga = genetic_algorithm(model, X_test, X_raw, feature_cols, sensitive_cols, n_samples)

        rs_scores.append(rs)
        ga_scores.append(ga)

        # who won this trial
        if ga > rs:
            winner = "GA wins"
        elif rs > ga:
            winner = "RS wins"
        else:
            winner = "Tie"

        print(f"  trial {trial+1:2d}:  RS={rs:.4f}   GA={ga:.4f}   "
              f"Delta={ga-rs:+.4f}   >> {winner}")

    # wilcoxon signed-rank test to check if difference is significant
    # non-parametric and paired so it fits our setup
    try:
        diffs = np.array(ga_scores) - np.array(rs_scores)
        p_value = wilcoxon(diffs).pvalue if len(set(diffs.round(6))) > 1 else 1.0
    except Exception:
        p_value = float("nan")

    rs_mean = float(np.mean(rs_scores))
    ga_mean = float(np.mean(ga_scores))

    if ga_mean > rs_mean:
        overall = "OVERALL WINNER: GA"
    elif rs_mean > ga_mean:
        overall = "OVERALL WINNER: RS (Baseline)"
    else:
        overall = "OVERALL: TIE"

    pv_str = f"{p_value:.4f}" if not np.isnan(p_value) else "nan"
    print(f"\n  RS: {rs_mean:.4f} (+/-{np.std(rs_scores):.4f})   "
          f"GA: {ga_mean:.4f} (+/-{np.std(ga_scores):.4f})   "
          f"Delta={ga_mean - rs_mean:+.4f} "
          f"({(ga_mean - rs_mean) / (rs_mean + 1e-9) * 100:+.1f}%)   "
          f"p={pv_str}   >> {overall}")

    return {
        "dataset":    ds_name,
        "sensitive":  sensitive_cols,
        "rs_mean":    rs_mean,
        "rs_std":     float(np.std(rs_scores)),
        "ga_mean":    ga_mean,
        "ga_std":     float(np.std(ga_scores)),
        "delta":      float(ga_mean - rs_mean),
        "pct_change": float((ga_mean - rs_mean) / (rs_mean + 1e-9) * 100),
        "p_value":    float(p_value) if not np.isnan(p_value) else None,
        "rs_trials":  rs_scores,
        "ga_trials":  ga_scores,
    }


# main
def main():
    N_SAMPLES = 1000   # number of pairs to evaluate per trial
    N_TRIALS  = 10    # repeat each experiment 10 times for statistical testing

    all_results = []

    for name, config in DATASETS.items():
        result = run_experiment(name, config, n_samples=N_SAMPLES, n_trials=N_TRIALS)
        if result:
            all_results.append(result)

    # print summary table
    print("\n\n" + "=" * 108)
    print(f"  {'Dataset':<14} {'RS mean':>9} {'RS std':>8} {'GA mean':>9} "
          f"{'GA std':>8} {'Delta':>8} {'% Change':>9} {'p-value':>9}   Winner")
    print("=" * 108)

    for r in all_results:
        sig = " *" if r["p_value"] is not None and r["p_value"] < 0.05 else "  "
        pv  = f"{r['p_value']:.4f}" if r["p_value"] is not None else "   nan"

        if r["ga_mean"] > r["rs_mean"]:
            winner = "GA"
        elif r["rs_mean"] > r["ga_mean"]:
            winner = "RS (Baseline)"
        else:
            winner = "Tie"

        print(f"  {r['dataset']:<14} {r['rs_mean']:>9.4f} {r['rs_std']:>8.4f} "
              f"{r['ga_mean']:>9.4f} {r['ga_std']:>8.4f} "
              f"{r['delta']:>+8.4f} {r['pct_change']:>+8.1f}% "
              f"{pv:>9}{sig}   {winner}")

    print("=" * 108)
    print("  * p < 0.05 (Wilcoxon signed-rank, two-sided)\n")

    ga_wins = sum(1 for r in all_results if r["ga_mean"] > r["rs_mean"])
    rs_wins = sum(1 for r in all_results if r["rs_mean"] > r["ga_mean"])
    ties    = sum(1 for r in all_results if r["rs_mean"] == r["ga_mean"])
    print(f"  Final Score:  GA = {ga_wins} win(s)   RS = {rs_wins} win(s)   Ties = {ties}\n")

    # save results to DNN folder
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")
    os.makedirs(out_dir, exist_ok=True)

    # save summary json
    summary = [{k: v for k, v in r.items() if "trials" not in k} for r in all_results]
    with open(os.path.join(out_dir, "results_ga.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save per-trial data as csv
    rows = []
    for r in all_results:
        for t, (rs, ga) in enumerate(zip(r["rs_trials"], r["ga_trials"])):
            winner = "GA" if ga > rs else ("RS" if rs > ga else "Tie")
            rows.append({
                "dataset": r["dataset"],
                "trial":   t + 1,
                "rs_idi":  rs,
                "ga_idi":  ga,
                "delta":   ga - rs,
                "winner":  winner
            })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "trial_data_ga.csv"), index=False)
    print(f"  Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
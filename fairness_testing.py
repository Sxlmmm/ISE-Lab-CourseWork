import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

# Suppress some of the TF log spam - makes the console cleaner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Full list of datasets from the lab examples 
DATA_MAP = {
    "adult": {"p": "dataset/processed_adult.csv", "t": "Class-label", "s": ["race", "gender"]},
    "compas": {"p": "dataset/processed_compas.csv", "t": "Recidivism", "s": ["Race", "Sex"]},
    "german": {"p": "dataset/processed_german.csv", "t": "CREDITRATING", "s": ["PersonStatusSex", "AgeInYears"]},
    "credit": {"p": "dataset/processed_credit_with_numerical.csv", "t": "class", "s": ["SEX", "AGE"]},
    "law_school": {"p": "dataset/processed_law_school.csv", "t": "pass_bar", "s": ["race", "male"]},
    "communities_crime": {"p": "dataset/processed_communities_crime.csv", "t": "class", "s": ["Black"]},
    "dutch": {"p": "dataset/processed_dutch.csv", "t": "occupation", "s": ["sex"]},
    "kdd": {"p": "dataset/processed_kdd.csv", "t": "income", "s": ["race", "sex"]}
}

def load_and_prep(path, target):
    # Using pandas to load everything; manual conversion to float32 for TF compatibility
    data = pd.read_csv(path)
    X = data.drop(columns=[target]).values.astype('float32')
    y = data[target].values.astype('float32')
    
    # Standard split for coursework 
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
    
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)
    
    return xtrain, xtest, ytrain, ytest, list(data.drop(columns=[target]).columns), X

def build_dnn(input_dim):
    # Basic architecture: 64 -> 32 -> 1. Dropout added to improve generalisation 
    net = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return net

def get_fitness(model, indivs, s_indices, raw_data, gap=0.05):
    # Fitness is the difference between original prediction and counterfactual prediction 
    cfs = indivs.copy()
    for idx in s_indices:
        vals = np.unique(raw_data[:, idx])
        for i in range(len(cfs)):
            # Pick a different observed value for the sensitive column
            opts = vals[vals != cfs[i, idx]]
            if len(opts) > 0:
                cfs[i, idx] = np.random.choice(opts)
    
    preds = model.predict(indivs, verbose=0).flatten()
    cf_preds = model.predict(cfs, verbose=0).flatten()
    
    diffs = np.abs(preds - cf_preds)
    return diffs, diffs > gap

def run_ga(model, xtest, raw_x, feats, sens_feats, budget=1000):
    # Genetic Algorithm to find discriminatory inputs 
    s_idx = [i for i, f in enumerate(feats) if f in sens_feats]
    ns_idx = [i for i, f in enumerate(feats) if f not in sens_feats]
    
    # Standard deviations for non-sensitive features to guide mutation scaling
    stds = np.std(raw_x[:, ns_idx], axis=0) + 1e-7
    
    pop_sz = 50
    gens = (budget // pop_sz) - 1
    
    pop = xtest[np.random.choice(len(xtest), pop_sz)].copy()
    total_disc = 0
    
    for _ in range(gens + 1):
        fit, is_disc = get_fitness(model, pop, s_idx, raw_x)
        total_disc += np.sum(is_disc)
        
        next_pop = []
        # Elitism: keep the best performers [cite: 78]
        best_guys = np.argsort(fit)[-5:]
        for b in best_guys:
            next_pop.append(pop[b])
            
        while len(next_pop) < pop_sz:
            # Tournament selection (k=2)
            t1, t2 = np.random.choice(pop_sz, 2)
            p1 = pop[t1] if fit[t1] > fit[t2] else pop[t2]
            
            t3, t4 = np.random.choice(pop_sz, 2)
            p2 = pop[t3] if fit[t3] > fit[t4] else pop[t4]
            
            # Crossover non-sensitive genes
            child = p1.copy()
            mask = np.random.rand(len(ns_idx)) < 0.5
            for i, change in enumerate(mask):
                if change:
                    child[ns_idx[i]] = p2[ns_idx[i]]
            
            # Mutation with small Gaussian noise
            for i, col in enumerate(ns_idx):
                if np.random.rand() < 0.1:
                    child[col] += np.random.normal(0, 0.05 * stds[i])
            
            next_pop.append(child)
        pop = np.array(next_pop)
        
    return total_disc / (pop_sz * (gens + 1))

def run_baseline(model, xtest, raw_x, feats, sens_feats, budget=1000):
    # Random Search baseline as specified in Lab 4 
    s_idx = [i for i, f in enumerate(feats) if f in sens_feats]
    ns_idx = [i for i, f in enumerate(feats) if f not in sens_feats]
    stds = np.std(raw_x[:, ns_idx], axis=0) + 1e-7
    
    idxs = np.random.randint(0, len(xtest), budget)
    samples = xtest[idxs].copy()
    
    for i, col in enumerate(ns_idx):
        samples[:, col] += np.random.normal(0, 0.1 * stds[i], budget)
        
    _, is_disc = get_fitness(model, samples, s_idx, raw_x)
    return np.mean(is_disc)

if __name__ == "__main__":
    # Increased trials to 10 for better statistical significance 
    n_trials = 10
    all_trial_rows = []
    for dname, conf in DATA_MAP.items():
        if not os.path.exists(conf['p']):
            print(f"Skipping {dname}: file not found.")
            continue
            
        print(f"\n=== EVALUATING DATASET: {dname} ===")
        xtr, xte, ytr, yte, f_names, raw = load_and_prep(conf['p'], conf['t'])
        
        # Train model once per dataset
        m = build_dnn(xtr.shape[1])
        m.fit(xtr, ytr, epochs=20, batch_size=512, verbose=0)
        
        
        rs_scores, ga_scores = [], []
        for t in range(1, n_trials + 1):
            rs_val = run_baseline(m, xte, raw, f_names, conf['s'])
            ga_val = run_ga(m, xte, raw, f_names, conf['s'])
            
            delta = ga_val - rs_val
            winner = "GA" if ga_val > rs_val else "RS"
            
            # Store for CSV
            all_trial_rows.append({
                "dataset": dname,
                "trial": t,
                "rs_idi": round(rs_val, 4),
                "ga_idi": round(ga_val, 4),
                "delta": delta,
                "winner": winner
            })
            
            rs_scores.append(rs_val)
            ga_scores.append(ga_val)
            print(f" Trial {t}: RS={rs_val:.4f}, GA={ga_val:.4f}, Winner={winner}")
            
        # Quick summary stats in console
        _, p = wilcoxon(rs_scores, ga_scores)
        print(f"Done {dname}. Avg RS: {np.mean(rs_scores):.4f} , Avg GA: {np.mean(ga_scores):.4f}, p-value: {p:.4f}")

        #Export to CSV
        df = pd.DataFrame(all_trial_rows)
        df.to_csv('outputs/trial_data_ga.csv', index=False)
        print("\nFile saved to outputs/trial_data_ga.csv")
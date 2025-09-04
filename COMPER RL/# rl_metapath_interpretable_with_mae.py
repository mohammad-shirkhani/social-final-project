#!/usr/bin/env python3
# rl_metapath_interpretable_fixed.py
"""
Comprehensive solution for constant reward issue:
1. DFS path discovery for better pattern diversity
2. Enhanced reward shaping with pattern existence bonus
3. Guided exploration toward valid patterns
4. Improved state representation
5. Comprehensive debugging outputs
"""

import os
import math
import json
import pickle
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------- CONFIG -------------------------
DATA_DIR = r"C:\Users\jalal\Downloads\COMPER\ml-latest-small"
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "dueling_dqn_project_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Precompute / graph options
SUBSET_MODE = True
SUBSET_NUM_USERS = 200

MAX_PATTERN_LEN = 4
TOP_K_PATHS = 10  # Increased to capture more patterns
MAX_VISITS_PER_USER = 50000  # Increased to explore more paths

PATHS_CACHE = os.path.join(OUTPUT_DIR, "paths_cache_userwise.pkl")  # New cache
NUM_WORKERS_PRECOMP = 0  # 0 = serial (Windows-friendly)

# Memory-safe ItemSim
ITEMSIM_K = 15  # Increased neighborhood size
ADD_ITEMSIM = True

# RL / Network
REL_EMB_DIM = 32
PREFIX_GRU_HID = 64
STATE_EXTRA_DIM = 4  # [coverage_prefix, avg_paths_per_pair, depth_norm, pattern_exists]
STATE_DIM = PREFIX_GRU_HID + STATE_EXTRA_DIM

NUM_EPISODES = 300
REPLAY_CAPACITY = 12000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END   = 0.1  # Maintain more exploration
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 50
WARMUP_STEPS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation/sample sizes
TRAIN_SAMPLE_SIZE = 500
VAL_SUBSET_SIZE   = 200

# Reward weights
WEIGHT_COVERAGE_DELTA = 1.0
WEIGHT_SIMPLICITY     = 0.1
WEIGHT_TERMINAL_COV   = 2.0
PATTERN_EXISTS_BONUS  = 0.5  # Reward for creating a valid pattern

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------- Human-readable relation text -------------------------
REL_TEXT: Dict[str, str] = {
    "ItemSim": "to a similar item (same genre)",
}

# ------------------------- Utilities -------------------------
def safe_regression_metrics(y_true, preds):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(preds, dtype=np.float64)
    mask = (~np.isnan(y)) & (~np.isnan(p))
    if mask.sum() == 0:
        return float('inf'), float('inf')
    y, p = y[mask], p[mask]
    p = np.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        rmse = math.sqrt(mean_squared_error(y, p))
        mae = mean_absolute_error(y, p)
    except Exception:
        rmse, mae = float('inf'), float('inf')
    return rmse, mae

# ------------------------- Graph Construction -------------------------
def build_graph(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH):
    print("[INFO] Loading ratings and movies ...")
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)

    G = nx.DiGraph()

    # movieId -> title
    movieid_to_title = {}
    if {'movieId','title'}.issubset(movies.columns):
        for _, row in movies.iterrows():
            try:
                movieid_to_title[int(row['movieId'])] = str(row['title'])
            except:
                pass

    # rating edges
    for _, r in tqdm(ratings.iterrows(), total=len(ratings), desc="Adding rating edges", ncols=100):
        uid = f"u{int(r['userId'])}"
        iid = f"i{int(r['movieId'])}"
        G.add_node(uid, ntype='user')
        G.add_node(iid, ntype='item')
        rel_u2i = f"Rank{int(round(r['rating']))}"
        rel_i2u = f"BeRanked{int(round(r['rating']))}"
        G.add_edge(uid, iid, rel=rel_u2i)
        G.add_edge(iid, uid, rel=rel_i2u)

    # Enhanced ItemSim with more connections
    if ADD_ITEMSIM and 'genres' in movies.columns:
        print(f"[INFO] Adding ItemSim edges with K={ITEMSIM_K} per movie per genre")
        genre_to_movies = defaultdict(list)
        movie_to_genres = defaultdict(list)
        
        # Build genre mappings
        for _, row in movies.iterrows():
            node = f"i{int(row['movieId'])}"
            genres = [g for g in str(row['genres']).split('|') 
                     if g and g != "(no genres listed)"]
            for g in genres:
                genre_to_movies[g].append(node)
            movie_to_genres[node] = genres

        # Connect movies that share at least one genre
        for movie, genres in tqdm(movie_to_genres.items(), desc="Adding ItemSim", ncols=100):
            for genre in genres:
                # Connect to other movies in the same genre
                for other in genre_to_movies[genre]:
                    if movie != other:
                        # Add bidirectional connections
                        G.add_edge(movie, other, rel='ItemSim')
                        G.add_edge(other, movie, rel='ItemSim')

    return G, ratings, movieid_to_title

def build_relation_index(G: nx.DiGraph):
    node_relation_neighbors = {}
    for node in G.nodes():
        d = defaultdict(list)
        for nb in G.successors(node):
            rel = G[node][nb]['rel']
            d[rel].append(nb)
        node_relation_neighbors[node] = d
    rels = sorted({G[u][v]['rel'] for u, v in G.edges()})
    # fill REL_TEXT for Rank/BeRanked
    for r in rels:
        if r.startswith("Rank"):
            n = r.replace("Rank","")
            REL_TEXT[r] = f"rated {n}"
        elif r.startswith("BeRanked"):
            n = r.replace("BeRanked","")
            REL_TEXT[r] = f"was rated {n}"
        else:
            REL_TEXT.setdefault(r, r)
    return node_relation_neighbors, rels

# ------------------------- Path Precomputation -------------------------
def find_paths_dfs(G, node_relation_neighbors, source, max_len=MAX_PATTERN_LEN, top_k=TOP_K_PATHS, max_visits=MAX_VISITS_PER_USER):
    """DFS-based path finding to capture more diverse patterns"""
    results = defaultdict(list)
    stack = [(source, [source], set([source]), 0)]  # (node, path, visited, depth)
    visit_count = 0
    path_count = defaultdict(int)
    
    while stack and visit_count < max_visits:
        node, path, visited, depth = stack.pop()
        visit_count += 1
        
        # Record item paths
        if G.nodes[node].get('ntype') == 'item' and len(path) > 1:
            item = node
            if path_count[item] < top_k:
                results[item].append(list(path))
                path_count[item] += 1
        
        # Continue if we haven't reached max depth
        if depth < max_len:
            neighbors = []
            for rel, nbs in node_relation_neighbors.get(node, {}).items():
                for nb in nbs:
                    if nb not in visited:
                        neighbors.append((nb, rel))
            
            # Randomize neighbor order to increase diversity
            random.shuffle(neighbors)
            
            for nb, rel in neighbors:
                new_visited = visited | {nb}
                stack.append((nb, path + [nb], new_visited, depth + 1))
    
    return results

def precompute_paths_by_user(G, node_relation_neighbors, user_nodes, pairs_to_cover=None,
                             cache_path=PATHS_CACHE, max_len=MAX_PATTERN_LEN, top_k=TOP_K_PATHS,
                             max_visits_per_user=MAX_VISITS_PER_USER, partial_save_every=50, num_workers=0):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_cache = pickle.load(f)
        print(f"[INFO] Loaded cache: {len(all_cache)} entries")
    else:
        all_cache = {}

    target_items_per_user = None
    if pairs_to_cover:
        target_items_per_user = defaultdict(set)
        for (u,i) in pairs_to_cover:
            target_items_per_user[u].add(i)

    users = list(user_nodes)
    processed = 0
    for u in tqdm(users, desc="Precompute (serial)", ncols=120):
        if target_items_per_user and (u not in target_items_per_user):
            processed += 1; continue
        
        # Use DFS instead of BFS for better pattern diversity
        res_dict = find_paths_dfs(G, node_relation_neighbors, u, max_len=max_len, top_k=top_k, max_visits=max_visits_per_user)
        
        if target_items_per_user:
            for item in target_items_per_user.get(u, []):
                paths = res_dict.get(item, [])
                if paths:
                    all_cache[(u,item)] = paths
        else:
            for item, paths in res_dict.items():
                if paths:
                    all_cache[(u,item)] = paths
        processed += 1
        if processed % partial_save_every == 0:
            with open(cache_path, "wb") as f:
                pickle.dump(all_cache, f)
    with open(cache_path, "wb") as f:
        pickle.dump(all_cache, f)
    print(f"[INFO] Precompute done. Cached entries: {len(all_cache)} -> {cache_path}")
    return all_cache

# ------------------------- Prefix Encoder -------------------------
class PrefixEncoder(nn.Module):
    def __init__(self, rel_vocab, rel_emb_dim=REL_EMB_DIM, gru_hid=PREFIX_GRU_HID):
        super().__init__()
        self.rel2idx = {r:i for i,r in enumerate(rel_vocab)}
        self.idx2rel = {i:r for r,i in self.rel2idx.items()}
        self.unknown_idx = len(self.rel2idx)
        self.rel_emb = nn.Embedding(len(self.rel2idx)+1, rel_emb_dim, padding_idx=self.unknown_idx)
        self.gru = nn.GRU(rel_emb_dim, gru_hid, batch_first=True)
        self.max_len = MAX_PATTERN_LEN
        self.gru_hid = gru_hid

    def encode_batch(self, prefixes: List[List[str]]):
        B = len(prefixes)
        seq = torch.full((B, self.max_len), fill_value=self.unknown_idx, dtype=torch.long, device=DEVICE)
        for i, pref in enumerate(prefixes):
            for j, rel in enumerate(pref[:self.max_len]):
                seq[i,j] = self.rel2idx.get(rel, self.unknown_idx)
        emb = self.rel_emb(seq)
        _, h = self.gru(emb)
        return h.squeeze(0)  # [B, hid]

# ------------------------- Dueling DQN -------------------------
class DuelingDQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, action_dim)
        )
    def forward(self, x):
        h = self.shared(x)
        V = self.value_stream(h)
        A = self.adv_stream(h)
        return V + (A - A.mean(dim=1, keepdim=True))
    def value_and_advantage(self, x):
        h = self.shared(x)
        return self.value_stream(h), self.adv_stream(h)

# ------------------------- Replay Buffer -------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    def push(self, s,a,r,ns,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s,a,r,ns,done))
        else:
            self.buffer[self.pos] = (s,a,r,ns,done)
            self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch):
        batch = min(batch, len(self.buffer))
        s,a,r,ns,d = zip(*random.sample(self.buffer, batch))
        return np.stack(s), np.array(a), np.array(r), np.stack(ns), np.array(d)
    def __len__(self):
        return len(self.buffer)

# ------------------------- Pattern Environment -------------------------
class PatternEnv:
    def __init__(self, rel_vocab, all_paths_cache, sample_pairs):
        self.rel_vocab = rel_vocab
        self.rel_idx = {r:i for i,r in enumerate(rel_vocab)}
        self.idx2rel = {i:r for r,i in self.rel_idx.items()}
        self.max_len = MAX_PATTERN_LEN
        self.all_paths_cache = all_paths_cache
        self.sample_pairs = list(sample_pairs)

        # Build pattern existence index
        self.pattern_exists = defaultdict(bool)
        self._rel_paths = {}
        print("[INFO] Building pattern existence index...")
        for key, paths in tqdm(all_paths_cache.items(), total=len(all_paths_cache), ncols=100):
            rel_paths = []
            for p in paths:
                rels = tuple(G[a][b]['rel'] for a,b in zip(p[:-1], p[1:]))
                rel_paths.append(rels)
                # Mark all prefixes as existing
                for i in range(1, len(rels)+1):
                    self.pattern_exists[rels[:i]] = True
            self._rel_paths[key] = rel_paths

    def reset(self):
        self.prefix = []
        self.step_count = 0
        self.last_cov = 0.0
        return self._state()

    def _coverage_of_prefix(self, prefix):
        """Calculate coverage for paths starting with prefix"""
        prefix_tuple = tuple(prefix)
        if not prefix:
            # Empty prefix covers all pairs
            return 1.0, 1.0
        
        cnt = 0
        for key in self.sample_pairs:
            rel_paths = self._rel_paths.get(key, [])
            for rels in rel_paths:
                if len(rels) >= len(prefix_tuple) and rels[:len(prefix_tuple)] == prefix_tuple:
                    cnt += 1
                    break
        
        cov = cnt / len(self.sample_pairs) if self.sample_pairs else 0.0
        return cov, cov  # Return coverage and same for simplicity

    def _state(self):
        cov, _ = self._coverage_of_prefix(self.prefix)
        depth_norm = len(self.prefix) / self.max_len
        pattern_exists = float(self.pattern_exists.get(tuple(self.prefix), False))
        avg_paths = 1.0  # Placeholder, not used in new reward
        
        extras = np.array([cov, avg_paths, depth_norm, pattern_exists], dtype=np.float32)
        return (list(self.prefix), extras)

    def step(self, action_idx):
        rel = self.idx2rel[action_idx]
        self.prefix.append(rel)
        self.step_count += 1
        
        # Calculate new coverage
        cov, _ = self._coverage_of_prefix(self.prefix)
        cov_delta = cov - self.last_cov
        self.last_cov = cov
        
        # Calculate simplicity reward
        sim = (self.max_len - len(self.prefix) + 1) / self.max_len
        
        # Check pattern existence
        pattern_exists = self.pattern_exists.get(tuple(self.prefix), False)
        exists_bonus = PATTERN_EXISTS_BONUS if pattern_exists else 0.0
        
        # Combine rewards
        reward = (
            WEIGHT_COVERAGE_DELTA * max(0, cov_delta) + 
            WEIGHT_SIMPLICITY * sim + 
            exists_bonus
        )
        
        done = (self.step_count >= self.max_len)
        
        # Add terminal reward
        if done:
            t_rew, _, _ = self.terminal_reward()
            reward += t_rew
            
        return self._state(), reward, done, {}

    def terminal_reward(self):
        """Calculate reward for exact pattern match"""
        cnt = 0
        prefix_tuple = tuple(self.prefix)
        for key in self.sample_pairs:
            for rels in self._rel_paths.get(key, []):
                if rels == prefix_tuple:
                    cnt += 1
                    break
        
        cov = cnt / len(self.sample_pairs) if self.sample_pairs else 0.0
        rew = WEIGHT_TERMINAL_COV * cov
        return rew, cov, 0.0

# ------------------------- Training Loop -------------------------
def pattern_to_text(pattern): 
    return " → ".join(REL_TEXT.get(r, r) for r in pattern)

def path_to_text_with_titles(path, movieid_to_title):
    parts = []
    for a,b in zip(path[:-1], path[1:]):
        rel = G[a][b]['rel']; txt = REL_TEXT.get(rel, rel)
        if isinstance(b,str) and b.startswith("i"):
            try:
                mid = int(b[1:]); title = movieid_to_title.get(mid, f"movie_{mid}")
                parts.append(f"{txt} ({title})"); continue
            except: pass
        parts.append(txt)
    return " → ".join(parts)

def story_print_top(prefix_list, all_paths_cache, movieid_to_title, top_n=5):
    for rank,(pref, reward, cov, sim) in enumerate(prefix_list[:top_n], start=1):
        print("\n" + "="*60)
        print(f"Top #{rank} | Reward={reward:.4f} | Coverage={cov:.3f} | Simplicity={sim:.3f}")
        print(f"Pattern: {pattern_to_text(pref)}")
        exemplars = []
        for (u,i), paths in all_paths_cache.items():
            for p in paths:
                rels = [G[a][b]['rel'] for a,b in zip(p[:-1], p[1:])]
                if rels == pref:
                    exemplars.append((u,i,p)); break
            if len(exemplars) >= 3: break
        if not exemplars:
            print("  (No exemplar paths found.)"); continue
        print("  Example stories:")
        for k,(u,i,p) in enumerate(exemplars, start=1):
            print(f"   - Example {k}: User {u} → {path_to_text_with_titles(p, movieid_to_title)} → Item {i}")
    print("\n" + "="*60 + "\n")

def backoff_predictions_for_prefix(prefix, all_paths_cache, train_pairs_sample, val_pairs_subset, global_train_mean):
    """Predict using prefix match with fallback to global mean"""
    # Collect ratings from train for prefix match
    train_vals = []
    for (u,i,r) in train_pairs_sample:
        for p in all_paths_cache.get((u,i), []):
            rels = [G[a][b]['rel'] for a,b in zip(p[:-1], p[1:])]
            if len(rels) >= len(prefix) and rels[:len(prefix)] == prefix:
                train_vals.append(r); break
    pattern_mean = float(np.mean(train_vals)) if train_vals else global_train_mean

    # Predict on val
    y_true, y_pred = [], []
    for (u,i,r) in val_pairs_subset:
        y_true.append(r)
        matched = False
        for p in all_paths_cache.get((u,i), []):
            rels = [G[a][b]['rel'] for a,b in zip(p[:-1], p[1:])]
            if len(rels) >= len(prefix) and rels[:len(prefix)] == prefix:
                matched = True; break
        y_pred.append(pattern_mean if matched else global_train_mean)
    return np.array(y_true, dtype=float), np.array(y_pred, dtype=float)

def train_dueling_dqn(rel_vocab, all_paths_cache, sample_pairs_for_reward, val_pairs_subset, train_pairs_sample, global_train_mean, movieid_to_title):
    action_dim = len(rel_vocab)
    encoder = PrefixEncoder(rel_vocab).to(DEVICE)
    policy_net = DuelingDQNNet(STATE_DIM, action_dim).to(DEVICE)
    target_net = DuelingDQNNet(STATE_DIM, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(list(policy_net.parameters()) + list(encoder.parameters()), lr=LR)
    buffer = ReplayBuffer(REPLAY_CAPACITY)
    env = PatternEnv(rel_vocab, all_paths_cache, sample_pairs_for_reward)
    epsilon = EPS_START

    episode_rewards, rmse_history, mae_history = [], [], []
    best_prefixes, prefixes_seen = [], set()

    # Log some sample paths for debugging
    sample_key = next(iter(all_paths_cache.keys()))
    sample_paths = all_paths_cache[sample_key][:3]
    print("\n[DEBUG] Sample paths from cache:")
    for path in sample_paths:
        rels = [G[path[i]][path[i+1]]['rel'] for i in range(len(path)-1)]
        print(f"  Path: {rels}")
    print(f"Relation vocab: {rel_vocab}\n")

    pbar = trange(1, NUM_EPISODES+1, desc="RL Episodes", ncols=120)
    total_steps = 0
    for ep in pbar:
        prefix_state, extras = env.reset()
        done = False
        ep_reward = 0.0
        ep_log = []

        while not done:
            # Encode current state
            enc = encoder.encode_batch([prefix_state])                   
            feats = torch.tensor(extras, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            state_vec = torch.cat([enc, feats], dim=1)

            # Enhanced exploration: pattern-based epsilon
            pattern_exists = extras[-1] > 0.5
            effective_eps = epsilon * (0.5 if pattern_exists else 1.5)
            
            # Select action
            if random.random() < effective_eps:
                # Prefer actions that lead to existing patterns
                valid_actions = []
                for a_idx in range(action_dim):
                    test_prefix = prefix_state + [env.idx2rel[a_idx]]
                    if env.pattern_exists.get(tuple(test_prefix), False):
                        valid_actions.append(a_idx)
                
                if valid_actions:
                    a = random.choice(valid_actions)
                else:
                    a = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q = policy_net(state_vec).cpu().numpy()[0]
                    a = int(np.argmax(q))

            # Take action
            (ns_prefix, ns_extras), r, done, _ = env.step(a)
            ep_reward += r
            total_steps += 1
            ep_log.append((prefix_state, a, r))

            # Push to replay buffer
            s_enc  = enc.detach().cpu().numpy()[0]
            ns_enc = encoder.encode_batch([ns_prefix]).detach().cpu().numpy()[0]
            s_vec  = np.concatenate([s_enc , np.asarray(extras, dtype=np.float32)])
            ns_vec = np.concatenate([ns_enc, np.asarray(ns_extras, dtype=np.float32)])
            buffer.push(s_vec, a, r, ns_vec, float(done))

            # Update network
            if len(buffer) >= max(BATCH_SIZE, WARMUP_STEPS):
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(BATCH_SIZE)
                s_t  = torch.FloatTensor(s_b).to(DEVICE)
                a_t  = torch.LongTensor(a_b).unsqueeze(1).to(DEVICE)
                r_t  = torch.FloatTensor(r_b).unsqueeze(1).to(DEVICE)
                ns_t = torch.FloatTensor(ns_b).to(DEVICE)
                d_t  = torch.FloatTensor(d_b).unsqueeze(1).to(DEVICE)

                q_sa = policy_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    na = policy_net(ns_t).argmax(dim=1, keepdim=True)
                    q_ns = target_net(ns_t).gather(1, na)
                    target = r_t + GAMMA*q_ns*(1 - d_t)
                loss = F.mse_loss(q_sa, target)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            prefix_state, extras = ns_prefix, ns_extras

        # Validation with current pattern
        y_val, y_pred = backoff_predictions_for_prefix(
            prefix_state, all_paths_cache, 
            train_pairs_sample, val_pairs_subset, global_train_mean
        )
        rmse, mae = safe_regression_metrics(y_val, y_pred)
        rmse_history.append(rmse); mae_history.append(mae)

        # Bookkeeping
        episode_rewards.append(ep_reward)
        pattern_tuple = tuple(prefix_state)
        prefixes_seen.add(pattern_tuple)
        
        # Get terminal coverage
        _, cov_exact, _ = env.terminal_reward()
        best_prefixes.append((list(prefix_state), ep_reward, cov_exact, (MAX_PATTERN_LEN - len(prefix_state))/MAX_PATTERN_LEN))
        best_prefixes = sorted(best_prefixes, key=lambda x: x[1], reverse=True)[:200]

        # Update target network
        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(EPS_END, epsilon*EPS_DECAY)

        # Log episode details if reward > 0
        if ep_reward > 0.1:
            print(f"\n[EP{ep}] Reward: {ep_reward:.3f}, Cov: {cov_exact:.3f}, Pattern: {pattern_to_text(prefix_state)}")

        # Update progress bar
        pbar.set_postfix({
            "Ep": ep,
            "Reward": f"{ep_reward:.3f}",
            "Cov": f"{cov_exact:.3f}",
            "RMSE": f"{rmse:.4f}",
            "MAE": f"{mae:.4f}",
            "Eps": f"{epsilon:.3f}"
        })

        # Print top patterns every 50 episodes
        if ep % 50 == 0 or (ep_reward > 1.0 and cov_exact > 0):
            print(f"\n[INFO] Episode {ep}: Top-5 patterns")
            story_print_top(best_prefixes, all_paths_cache, movieid_to_title, top_n=5)

    trained = {
        "encoder": encoder,
        "policy_net": policy_net,
        "target_net": target_net,
        "episode_rewards": episode_rewards,
        "rmse_history": rmse_history,
        "mae_history": mae_history,
        "best_prefixes": best_prefixes,
        "rel_vocab": rel_vocab,
        "prefixes_seen": prefixes_seen
    }
    return trained

# ------------------------- Visualization -------------------------
def plot_rmse_mae(rmse_history, mae_history, outdir=OUTPUT_DIR):
    episodes = np.arange(1, len(rmse_history)+1)
    plt.figure(figsize=(10,5))
    plt.plot(episodes, rmse_history, label="RMSE")
    plt.plot(episodes, mae_history, label="MAE")
    plt.xlabel("Episode"); plt.ylabel("Error")
    plt.title("Validation RMSE & MAE over Episodes")
    plt.legend(); plt.grid(True); plt.tight_layout()
    path = os.path.join(outdir, "rmse_mae_over_episodes.png")
    plt.savefig(path); plt.close()
    print(f"[INFO] Saved {path}")

# ------------------------- Main Execution -------------------------
def main():
    global G, rel_vocab, node_relation_neighbors, movieid_to_title
    
    # Build graph
    print("[START] Building graph")
    G, ratings_df, movieid_to_title = build_graph(RATINGS_PATH, MOVIES_PATH)
    node_relation_neighbors, RELATIONS = build_relation_index(G)
    rel_vocab = RELATIONS
    print(f"[INFO] Relation vocab size: {len(rel_vocab)}")

    # Prepare pairs
    pairs_all = [(f"u{int(r['userId'])}", f"i{int(r['movieId'])}", float(r['rating'])) for _, r in ratings_df.iterrows()]
    random.shuffle(pairs_all)
    users = sorted({u for u,_,_ in pairs_all})[:SUBSET_NUM_USERS] if SUBSET_MODE else sorted({u for u,_,_ in pairs_all})
    pairs_set = {(u,i) for u,i,_ in pairs_all}

    # Precompute paths
    print("[START] Precomputing paths (resumeable)...")
    all_paths_cache = precompute_paths_by_user(
        G, node_relation_neighbors, users, pairs_to_cover=pairs_set,
        cache_path=PATHS_CACHE, max_len=MAX_PATTERN_LEN, top_k=TOP_K_PATHS,
        max_visits_per_user=MAX_VISITS_PER_USER, partial_save_every=50, num_workers=NUM_WORKERS_PRECOMP
    )

    # Split data
    train_pairs, val_pairs = train_test_split(pairs_all, test_size=0.15, random_state=SEED)
    train_pairs_sample = random.sample(train_pairs, min(TRAIN_SAMPLE_SIZE, len(train_pairs)))
    val_pairs_subset   = random.sample(val_pairs  , min(VAL_SUBSET_SIZE  , len(val_pairs)))
    global_train_mean  = float(np.mean([r for _,_,r in train_pairs_sample]))
    print(f"[INFO] TRAIN_SAMPLE_SIZE={len(train_pairs_sample)}, VAL_SUBSET_SIZE={len(val_pairs_subset)}, global_mean={global_train_mean:.3f}")

    # Train DQN
    print("[START] Training Dueling DQN")
    trained = train_dueling_dqn(
        rel_vocab=rel_vocab,
        all_paths_cache=all_paths_cache,
        sample_pairs_for_reward=list(all_paths_cache.keys())[:min(2000, len(all_paths_cache))],
        val_pairs_subset=val_pairs_subset,
        train_pairs_sample=train_pairs_sample,
        global_train_mean=global_train_mean,
        movieid_to_title=movieid_to_title
    )

    # Save results
    print("[INFO] Saving results...")
    plot_rmse_mae(trained['rmse_history'], trained['mae_history'], outdir=OUTPUT_DIR)

    # Save models
    torch.save(trained['policy_net'].state_dict(), os.path.join(OUTPUT_DIR, "policy_net_final.pth"))
    with open(os.path.join(OUTPUT_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(trained['encoder'].state_dict(), f)
    with open(os.path.join(OUTPUT_DIR, "best_prefixes.pkl"), "wb") as f:
        pickle.dump(trained['best_prefixes'], f)

    # Print final top patterns
    print("\n[FINAL] Top-5 discovered patterns:")
    story_print_top(trained['best_prefixes'], all_paths_cache, movieid_to_title, top_n=5)

    print("[DONE] All outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
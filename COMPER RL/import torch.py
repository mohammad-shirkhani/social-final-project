import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym
import warnings
import gc

# تنظیمات اولیه
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
torch.manual_seed(42)
np.random.seed(42)

# 1. بارگیری داده‌ها
print("📂 بارگیری داده‌ها...")
ratings = pd.read_csv('C:/Users/jalal/Downloads/COMPER/ml-latest-small/ratings.csv')
movies = pd.read_csv('C:/Users/jalal/Downloads/COMPER/ml-latest-small/movies.csv')
print(f"✅ داده‌ها بارگیری شدند: {len(ratings)} ریتینگ، {len(movies)} فیلم")

# 2. محاسبه شباهت‌ها با رفع خطا
print("🧮 ایجاد ماتریس امتیازدهی...")
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

print("👥 محاسبه شباهت کاربران...")
# راه حل جایگزین برای محاسبه شباهت کاربران
user_corr = rating_matrix.T.corr(method='pearson')
user_corr.index.name = 'userA'
user_corr.columns.name = 'userB'
user_sim = user_corr.stack().reset_index(name='correlation')
user_sim = user_sim[
    (user_sim['correlation'] > 0.5) & 
    (user_sim['userA'] != user_sim['userB']) &
    (user_sim['userA'] < user_sim['userB'])
].reset_index(drop=True)
print(f"✅ {len(user_sim)} جفت کاربر مشابه")

print("🎬 محاسبه شباهت آیتم‌ها...")
# راه حل جایگزین برای محاسبه شباهت آیتم‌ها
item_corr = rating_matrix.corr(method='pearson')
item_corr.index.name = 'itemA'
item_corr.columns.name = 'itemB'
item_sim = item_corr.stack().reset_index(name='correlation')
item_sim = item_sim[
    (item_sim['correlation'] > 0.5) & 
    (item_sim['itemA'] != item_sim['itemB']) &
    (item_sim['itemA'] < item_sim['itemB'])
].reset_index(drop=True)
print(f"✅ {len(item_sim)} جفت فیلم مشابه")

# 3. ساخت گراف همکاری
print("🕸️ ساخت گراف همکاری...")
G = nx.Graph()

# افزودن گره‌ها
user_nodes = ratings['userId'].unique()
item_nodes = ratings['movieId'].unique()
G.add_nodes_from(user_nodes, node_type='user')
G.add_nodes_from(item_nodes, node_type='item')
print(f"✅ {len(user_nodes)} کاربر و {len(item_nodes)} فیلم به گراف اضافه شدند")

# افزودن یال‌های امتیاز
for _, row in tqdm(ratings.iterrows(), total=len(ratings), desc="یال‌های امتیاز"):
    G.add_edge(row['userId'], row['movieId'], 
               relation_type=f'Rated_{int(row["rating"])}',
               weight=row['rating'])

# افزودن یال‌های شباهت کاربران
if not user_sim.empty:
    for _, row in tqdm(user_sim.iterrows(), total=len(user_sim), desc="شباهت کاربران"):
        G.add_edge(row['userA'], row['userB'], 
                   relation_type='UserSim',
                   weight=row['correlation'])

# افزودن یال‌های شباهت آیتم‌ها
if not item_sim.empty:
    for _, row in tqdm(item_sim.iterrows(), total=len(item_sim), desc="شباهت آیتم‌ها"):
        G.add_edge(row['itemA'], row['itemB'], 
                   relation_type='ItemSim',
                   weight=row['correlation'])

print(f"✅ گراف ساخته شد: {G.number_of_nodes()} گره، {G.number_of_edges()} یال")

# 4. استخراج مسیرهای متا بهینه‌سازی شده
def extract_meta_paths(G, user, item, max_paths=5, max_length=3):
    """استخراج مسیرهای متا بین کاربر و آیتم بهینه‌سازی شده با BFS"""
    if user not in G or item not in G:
        return []
    
    try:
        # استفاده از BFS برای یافتن مسیرها با محدودیت تعداد
        paths = []
        queue = deque([(user, [user])])
        visited = set()
        
        while queue and len(paths) < max_paths:
            node, path = queue.popleft()
            visited.add(node)
            
            # اگر به آیتم رسیدیم و مسیر معتبر است
            if node == item and len(path) > 1:
                paths.append(path)
                continue
                
            # اگر طول مسیر از حد مجاز بیشتر شد
            if len(path) >= max_length:
                continue
                
            # بررسی همسایه‌ها
            for neighbor in G.neighbors(node):
                # جلوگیری از حلقه
                if neighbor not in path and neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return paths
    except Exception as e:
        print(f"خطا در استخراج مسیر: {e}")
        return []

# 5. مدل COMPER
class COMPER(nn.Module):
    def __init__(self, num_users, num_items, num_relations, emb_dim=64, lstm_dim=64):
        super(COMPER, self).__init__()
        
        # لایه‌های جاسازی
        self.user_emb = nn.Embedding(num_users + 1, emb_dim)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        self.node_type_emb = nn.Embedding(2, emb_dim)  # 0: user, 1: item
        
        # لایه LSTM
        self.lstm = nn.LSTM(3 * emb_dim, lstm_dim, batch_first=True)
        
        # مکانیزم توجه
        self.attention = nn.Sequential(
            nn.Linear(lstm_dim + 2 * emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # لایه پیش‌بینی
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user, item, paths):
        # جاسازی کاربر و آیتم
        u_emb = self.user_emb(user)
        i_emb = self.item_emb(item)
        
        path_embs = []
        for path in paths:
            seq = []
            for i in range(len(path) - 1):
                node = path[i]
                next_node = path[i+1]
                
                # جاسازی گره
                if G.nodes[node].get('node_type', '') == 'user':
                    node_emb = self.user_emb(torch.tensor([node], device=user.device))
                    node_type = torch.tensor([0], device=user.device)
                else:
                    node_emb = self.item_emb(torch.tensor([node], device=user.device))
                    node_type = torch.tensor([1], device=user.device)
                
                # جاسازی رابطه
                rel_type = G.edges[node, next_node].get('relation_type', 'Rated_3')
                rel_idx = relation2idx.get(rel_type, 0)
                rel_emb = self.relation_emb(torch.tensor([rel_idx], device=user.device))
                
                # ترکیب ویژگی‌ها
                features = torch.cat([node_emb, self.node_type_emb(node_type), rel_emb], dim=-1)
                seq.append(features)
            
            if not seq:
                continue
                
            # پردازش با LSTM
            seq = torch.stack(seq).unsqueeze(0)
            _, (h_n, _) = self.lstm(seq)
            path_emb = h_n[-1]
            path_embs.append(path_emb)
        
        if not path_embs:
            return torch.zeros(1, device=user.device), torch.zeros(1, device=user.device)
            
        path_embs = torch.cat(path_embs, dim=0)
        context = torch.cat([u_emb, i_emb], dim=-1).repeat(len(path_embs), 1)
        att_input = torch.cat([path_embs, context], dim=1)
        att_scores = self.attention(att_input)
        att_weights = F.softmax(att_scores, dim=0)
        
        # تجمع با توجه
        aggregated = torch.sum(att_weights * path_embs, dim=0)
        
        # پیش‌بینی
        prediction = self.fc(aggregated)
        return prediction, att_weights

# 6. محیط یادگیری تقویتی
class RecSysEnv(gym.Env):
    def __init__(self, model, data, G, relation2idx):
        super(RecSysEnv, self).__init__()
        self.model = model
        self.data = data
        self.G = G
        self.relation2idx = relation2idx
        self.current_idx = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,))
        self.action_space = spaces.MultiBinary(5)
        self.max_steps = min(500, len(data))
        self.current_step = 0
        
    def reset(self):
        self.current_idx = 0
        self.current_step = 0
        return self._get_state()
    
    def step(self, action):
        self.current_step += 1
        row = self.data.iloc[self.current_idx]
        user = torch.tensor([row['userId']], device=device)
        item = torch.tensor([row['movieId']], device=device)
        true_rating = row['rating']
        
        paths = extract_meta_paths(self.G, user.item(), item.item(), max_paths=5, max_length=3)
        selected_paths = [p for i, p in enumerate(paths) if i < len(action) and action[i] == 1]
        
        if not selected_paths and paths:
            selected_paths = paths[:3]
        
        # پیش‌بینی مدل
        with torch.no_grad():
            try:
                pred, _ = self.model(user, item, selected_paths)
                rmse = torch.sqrt(F.mse_loss(pred, torch.tensor([true_rating], device=device))).item()
            except:
                rmse = 3.0
        
        # محاسبه پاداش
        sid = self.compute_sid(selected_paths) if selected_paths else 0.0
        reward = -rmse + 0.5 * sid
        
        # به‌روزرسانی ایندکس
        self.current_idx = (self.current_idx + 1) % len(self.data)
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        row = self.data.iloc[self.current_idx]
        user = torch.tensor([row['userId']], device=device)
        item = torch.tensor([row['movieId']], device=device)
        with torch.no_grad():
            u_emb = self.model.user_emb(user).cpu().numpy().flatten()
            i_emb = self.model.item_emb(item).cpu().numpy().flatten()
        return np.concatenate([u_emb, i_emb])
    
    def compute_sid(self, paths):
        pattern_counts = {}
        for path in paths:
            pattern = []
            for i in range(len(path) - 1):
                rel = self.G.edges[path[i], path[i+1]].get('relation_type', 'Rated_3')[:3]
                pattern.append(rel)
            pattern_str = '-'.join(pattern)
            pattern_counts[pattern_str] = pattern_counts.get(pattern_str, 0) + 1
        
        n = len(paths)
        sid = 1 - sum([count*(count-1) for count in pattern_counts.values()]) / (n*(n-1)) if n > 1 else 0
        return sid

# 7. آماده‌سازی داده‌ها
print("⚙️ آماده‌سازی مدل...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ دستگاه: {device}")

# پارامترهای مدل
num_users = ratings['userId'].nunique()
num_items = ratings['movieId'].nunique()
relation_types = list(set(d['relation_type'] for _, _, d in G.edges(data=True)))
relation2idx = {rel: idx for idx, rel in enumerate(relation_types)}

# تقسیم داده‌ها
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
print(f"✂️ تقسیم داده: آموزش={len(train_data)}, تست={len(test_data)}")

# ایجاد مدل
model = COMPER(num_users, num_items, len(relation_types), emb_dim=32, lstm_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. آموزش مدل پایه با بهینه‌سازی
print("🔥 آموزش مدل پایه...")
train_data_small = train_data.sample(frac=0.1, random_state=42)  # کاهش حجم داده‌ها
print(f"🔍 آموزش با {len(train_data_small)} نمونه (10% داده‌ها)")

for epoch in range(1):  # فقط یک دوره برای تست
    epoch_loss = 0
    no_path_count = 0
    error_count = 0
    processed_count = 0
    
    progress_bar = tqdm(train_data_small.iterrows(), total=len(train_data_small), desc=f"دوره {epoch+1}/1")
    for idx, row in progress_bar:
        try:
            user = torch.tensor([row['userId']], device=device)
            item = torch.tensor([row['movieId']], device=device)
            true_rating = torch.tensor([row['rating']], dtype=torch.float, device=device)
            
            # استخراج مسیرها با محدودیت
            paths = extract_meta_paths(G, user.item(), item.item(), max_paths=5, max_length=3)
            
            if not paths:
                no_path_count += 1
                continue
                
            # آموزش مدل
            pred, _ = model(user, item, paths)
            loss = F.mse_loss(pred, true_rating)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            processed_count += 1
            
            # به‌روزرسانی نوار پیشرفت
            if idx % 100 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    paths=len(paths),
                    processed=f"{processed_count}/{len(train_data_small)}"
                )
            
        except Exception as e:
            error_count += 1
            # print(f"Error: {e}")  # فعال‌سازی برای دیباگ
    
    # محاسبه میانگین خطا
    valid_samples = processed_count
    if valid_samples > 0:
        avg_loss = epoch_loss / valid_samples
    else:
        avg_loss = 0
        
    print(f"\nدوره {epoch+1} - میانگین خطا: {avg_loss:.4f}")
    print(f"نمونه‌های بدون مسیر: {no_path_count}, نمونه‌های با خطا: {error_count}")
    
    # آزادسازی حافظه
    gc.collect()
    torch.cuda.empty_cache()

# 9. آموزش عامل RL
print("🤖 آموزش عامل RL...")
env = RecSysEnv(model, train_data_small, G, relation2idx)
env = DummyVecEnv([lambda: env])

rl_model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.0003,
    n_steps=256,
    batch_size=32,
    n_epochs=3,
    device=device
)

# آموزش با نوار پیشرفت
total_timesteps = 1000
progress_bar = tqdm(total=total_timesteps, desc="آموزش RL")
for i in range(total_timesteps // 100):
    rl_model.learn(total_timesteps=100)
    progress_bar.update(100)
progress_bar.close()

# 10. ارزیابی مدل
print("🧪 ارزیابی مدل...")
def evaluate(model, data):
    predictions, truths = [], []
    no_path_count = 0
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc="ارزیابی"):
        user = torch.tensor([row['userId']], device=device)
        item = torch.tensor([row['movieId']], device=device)
        true_rating = row['rating']
        
        # استخراج مسیرها با محدودیت
        paths = extract_meta_paths(G, user.item(), item.item(), max_paths=5, max_length=3)
        
        if not paths:
            no_path_count += 1
            predictions.append(3.0)  # مقدار پیش‌فرض
            truths.append(true_rating)
            continue
            
        with torch.no_grad():
            pred, _ = model(user, item, paths)
            predictions.append(pred.item())
            truths.append(true_rating)
    
    print(f"نمونه‌های بدون مسیر در ارزیابی: {no_path_count}")
    return (
        np.sqrt(mean_squared_error(truths, predictions)),
        mean_absolute_error(truths, predictions)
    )

# ارزیابی با زیرمجموعه کوچک
test_data_small = test_data.sample(frac=0.1, random_state=42)
test_rmse, test_mae = evaluate(model, test_data_small)
print(f"✅ نتایج: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

# 11. تولید توضیح
print("\n🎭 تولید نمونه توضیح:")
sample = test_data_small.sample(1).iloc[0]
user_id = sample['userId']
item_id = sample['movieId']
movie_title = movies[movies['movieId'] == item_id]['title'].values[0]
true_rating = sample['rating']

print(f"👤 کاربر: {user_id}, 🎬 فیلم: {movie_title}, ⭐ امتیاز واقعی: {true_rating}")

# استخراج مسیرها
paths = extract_meta_paths(G, user_id, item_id, max_paths=10, max_length=3)
if not paths:
    print("هیچ مسیر تفسیری یافت نشد")
else:
    user_tensor = torch.tensor([user_id], device=device)
    item_tensor = torch.tensor([item_id], device=device)
    pred, att_weights = model(user_tensor, item_tensor, paths)
    
    print(f"🔮 امتیاز پیش‌بینی شده: {pred.item():.2f}")
    
    # نمایش 3 مسیر برتر
    weights = att_weights.squeeze().cpu().detach().numpy()
    top_indices = weights.argsort()[-3:][::-1] if len(weights) > 3 else range(len(weights))
    
    print("\n💡 مسیرهای تفسیری برتر:")
    for i, idx in enumerate(top_indices, 1):
        if idx >= len(paths):
            continue
            
        path = paths[idx]
        weight = weights[idx]
        explanation = []
        
        for j in range(len(path) - 1):
            from_node = path[j]
            to_node = path[j+1]
            rel = G.edges[from_node, to_node]['relation_type']
            
            if 'UserSim' in rel:
                explanation.append(f"کاربر مشابه {from_node}")
            elif 'ItemSim' in rel:
                movie_title = movies[movies['movieId'] == from_node]['title'].values[0] if from_node in movies['movieId'].values else f"فیلم {from_node}"
                explanation.append(f"فیلم مشابه '{movie_title}'")
            elif 'Rated' in rel:
                rating = rel.split('_')[-1]
                if G.nodes[from_node].get('node_type', '') == 'user':
                    explanation.append(f"کاربر {from_node} → امتیاز {rating}")
                else:
                    movie_title = movies[movies['movieId'] == from_node]['title'].values[0] if from_node in movies['movieId'].values else f"فیلم {from_node}"
                    explanation.append(f"'{movie_title}' → امتیاز {rating}")
        
        print(f"{i}. {' → '.join(explanation)} (وزن: {weight:.4f})")
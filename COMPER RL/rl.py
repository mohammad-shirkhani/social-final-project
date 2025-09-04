import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import warnings
import json
import math

# تنظیمات اولیه
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# 1. بارگیری داده‌ها
print("📂 بارگیری داده‌ها...")
ratings = pd.read_csv('C:/Users/jalal/Downloads/COMPER/ml-latest-small/ratings.csv')
movies = pd.read_csv('C:/Users/jalal/Downloads/COMPER/ml-latest-small/movies.csv')
print(f"✅ داده‌ها بارگیری شدند: {len(ratings)} ریتینگ، {len(movies)} فیلم")

# 2. ایجاد نگاشت شناسه‌ها به اندیس‌های پیوسته
print("🔄 ایجاد نگاشت شناسه‌ها...")
user_ids = ratings['userId'].unique()
user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_ids = ratings['movieId'].unique()
item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

# افزودن ستون‌های جدید با اندیس‌های پیوسته
ratings['user_idx'] = ratings['userId'].map(user_to_idx)
ratings['item_idx'] = ratings['movieId'].map(item_to_idx)

print(f"🔢 {len(user_ids)} کاربر و {len(item_ids)} فیلم نگاشت شدند")

# 3. محاسبه شباهت‌ها با اندیس‌های پیوسته
print("🧮 ایجاد ماتریس امتیازدهی...")
rating_matrix = ratings.pivot(index='user_idx', columns='item_idx', values='rating').fillna(0)

print("👥 محاسبه شباهت کاربران...")
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

# 4. ساخت گراف همکاری با اندیس‌های پیوسته
print("🕸️ ساخت گراف همکاری...")
G = nx.Graph()

# افزودن گره‌ها
user_nodes = ratings['user_idx'].unique()
item_nodes = ratings['item_idx'].unique()
G.add_nodes_from(user_nodes, node_type='user')
G.add_nodes_from(item_nodes, node_type='item')
print(f"✅ {len(user_nodes)} کاربر و {len(item_nodes)} فیلم به گراف اضافه شدند")

# افزودن یال‌های امتیاز
for _, row in tqdm(ratings.iterrows(), total=len(ratings), desc="یال‌های امتیاز"):
    G.add_edge(row['user_idx'], row['item_idx'], 
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

# 5. استخراج مسیرهای متا بهینه‌سازی شده
def extract_meta_paths(G, user_idx, item_idx, max_paths=10, max_length=4):
    """استخراج مسیرهای متا بین کاربر و آیتم بهینه‌سازی شده با BFS"""
    if user_idx not in G or item_idx not in G:
        return []
    
    try:
        paths = []
        queue = deque([(user_idx, [user_idx])])
        visited = set()
        
        while queue and len(paths) < max_paths:
            node, path = queue.popleft()
            visited.add(node)
            
            if node == item_idx and len(path) > 1:
                paths.append(path)
                continue
                
            if len(path) >= max_length:
                continue
                
            for neighbor in G.neighbors(node):
                if neighbor not in path and neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return paths
    except Exception as e:
        print(f"خطا در استخراج مسیر: {e}")
        return []

# 6. مدل COMPER پیشرفته با قابلیت‌های تفسیرپذیری
class AdvancedCOMPER(nn.Module):
    def __init__(self, num_users, num_items, relation2idx, emb_dim=64, lstm_dim=64):
        super(AdvancedCOMPER, self).__init__()
        self.relation2idx = relation2idx
        
        # لایه‌های جاسازی
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.relation_emb = nn.Embedding(len(relation2idx), emb_dim)
        self.node_type_emb = nn.Embedding(2, emb_dim)  # 0: user, 1: item
        
        # لایه LSTM برای پردازش مسیرها
        self.lstm = nn.LSTM(3 * emb_dim, lstm_dim, batch_first=True)
        
        # مکانیزم توجه پویا
        self.attention = nn.Sequential(
            nn.Linear(lstm_dim + 2 * emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # لایه پیش‌بینی
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # لایه ایمنی برای محدودیت‌های یادگیری تقویتی
        self.safety_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # لایه ارزش و مزیت برای معماری Dueling DQN
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(lstm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_idx, item_idx, paths):
        # جاسازی کاربر و آیتم
        u_emb = self.user_emb(user_idx)
        i_emb = self.item_emb(item_idx)
        
        path_embs = []
        for path in paths:
            seq = []
            for i in range(len(path) - 1):
                node = path[i]
                next_node = path[i+1]
                
                # جاسازی گره
                if G.nodes[node].get('node_type', '') == 'user':
                    node_emb = self.user_emb(torch.tensor([node], device=user_idx.device, dtype=torch.long))
                    node_type = torch.tensor([0], device=user_idx.device, dtype=torch.long)
                else:
                    node_emb = self.item_emb(torch.tensor([node], device=user_idx.device, dtype=torch.long))
                    node_type = torch.tensor([1], device=user_idx.device, dtype=torch.long)
                
                # جاسازی رابطه
                rel_type = G.edges[node, next_node].get('relation_type', 'Rated_3')
                rel_idx = self.relation2idx.get(rel_type, 0)
                rel_emb = self.relation_emb(torch.tensor([rel_idx], device=user_idx.device, dtype=torch.long))
                
                # ترکیب ویژگی‌ها و حذف بعد اضافی
                node_emb = node_emb.squeeze(0)
                node_type_emb = self.node_type_emb(node_type).squeeze(0)
                rel_emb = rel_emb.squeeze(0)
                
                features = torch.cat([node_emb, node_type_emb, rel_emb], dim=-1)
                seq.append(features)
            
            if not seq:
                continue
                
            # ایجاد تانسور 3 بعدی مناسب برای LSTM
            seq_tensor = torch.stack(seq)  # شکل: [sequence_length, features]
            seq_tensor = seq_tensor.unsqueeze(0)  # شکل: [1, sequence_length, features] - 3 بعدی
            
            # پردازش با LSTM
            _, (h_n, _) = self.lstm(seq_tensor)
            path_emb = h_n[-1]  # آخرین hidden state از آخرین لایه
            path_embs.append(path_emb)
        
        if not path_embs:
            return torch.zeros(1, device=user_idx.device), torch.zeros(1, device=user_idx.device)
            
        path_embs = torch.cat(path_embs, dim=0)
        context = torch.cat([u_emb, i_emb], dim=-1).repeat(len(path_embs), 1)
        att_input = torch.cat([path_embs, context], dim=1)
        att_scores = self.attention(att_input)
        att_weights = F.softmax(att_scores, dim=0)
        
        # تجمع با توجه
        aggregated = torch.sum(att_weights * path_embs, dim=0)
        
        # پیش‌بینی با معماری Dueling
        value = self.value_stream(aggregated)
        advantage = self.advantage_stream(aggregated)
        q_value = value + (advantage - advantage.mean())
        
        # اعمال محدودیت‌های ایمنی
        safety = self.safety_layer(torch.cat([u_emb, i_emb], dim=-1))
        prediction = q_value * safety
        
        return prediction, att_weights

# 7. محیط یادگیری تقویتی پیشرفته برای سیستم توصیه‌گر
class AdvancedRecSysEnv(gym.Env):
    def __init__(self, model, data, G, relation2idx, feature_names):
        super(AdvancedRecSysEnv, self).__init__()
        self.model = model
        self.data = data
        self.G = G
        self.relation2idx = relation2idx
        self.feature_names = feature_names
        
        # تعریف فضای حالت و عمل
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,))  # افزایش به 128 بعد
        self.action_space = spaces.MultiBinary(5)  # انتخاب حداکثر 5 مسیر
        
        # تنظیمات ایمنی
        self.safety_constraints = {
            'min_rating': 1.0,
            'max_rating': 5.0,
            'min_paths': 1
        }
        
        # تنظیمات پاداش (اصلاح شده)
        self.reward_weights = {
            'accuracy': 2.0,  # افزایش اهمیت دقت
            'diversity': 0.3, # کاهش اهمیت تنوع
            'safety': 0.1,   # کاهش اهمیت ایمنی
            'simplicity': 0.05 # کاهش اهمیت سادگی
        }
        
        # ذخیره تاریخچه برای تفسیرپذیری
        self.explanation_history = []
        
    def reset(self):
        self.current_idx = 0
        self.current_step = 0
        self.selected_paths_history = []
        return self._get_state()
    
    def step(self, action):
        self.current_step += 1
        row = self.data.iloc[self.current_idx]
        user_idx = torch.tensor([row['user_idx']], device='cpu', dtype=torch.long)
        item_idx = torch.tensor([row['item_idx']], device='cpu', dtype=torch.long)
        true_rating = row['rating']
        
        # استخراج مسیرهای ممکن
        paths = extract_meta_paths(self.G, user_idx.item(), item_idx.item(), max_paths=10, max_length=4)
        
        # انتخاب مسیرها بر اساس عمل عامل
        selected_paths = [p for i, p in enumerate(paths) if i < len(action) and action[i] == 1]
        
        # اگر هیچ مسیری انتخاب نشده باشد، از اولین مسیرها استفاده می‌کنیم
        if not selected_paths and paths:
            selected_paths = paths[:3]
        
        # ذخیره تاریخچه انتخاب‌ها
        self.selected_paths_history.append({
            'user': user_idx.item(),
            'item': item_idx.item(),
            'selected_paths': selected_paths,
            'all_paths': paths
        })
        
        # پیش‌بینی امتیاز
        with torch.no_grad():
            try:
                pred, _ = self.model(user_idx, item_idx, selected_paths)
                rmse = torch.sqrt(F.mse_loss(pred, torch.tensor([true_rating], device='cpu'))).item()
                pred_rating = pred.item()
            except Exception as e:
                print(f"خطا در پیش‌بینی: {e}")
                rmse = 3.0
                pred_rating = 3.0
        
        # محاسبه معیارهای پاداش (اصلاح شده)
        reward_components = self._calculate_reward_components(
            selected_paths, true_rating, pred_rating, rmse
        )
        
        # محاسبه پاداش نهایی
        total_reward = sum(
            self.reward_weights[k] * reward_components[k] 
            for k in self.reward_weights
        )
        
        # ایجاد تفسیر برای این مرحله
        explanation = self._generate_explanation(user_idx, item_idx, selected_paths, true_rating, pred_rating)
        self.explanation_history.append(explanation)
        
        # به‌روزرسانی ایندکس
        self.current_idx = (self.current_idx + 1) % len(self.data)
        done = self.current_step >= 200  # افزایش مراحل
        
        # ذخیره نتایج برای محاسبه MAE و RMSE
        result = {
            'true_rating': true_rating,
            'pred_rating': pred_rating,
            'rmse': rmse,
            'mae': abs(true_rating - pred_rating)
        }
        
        return self._get_state(), total_reward, done, result
    
    def _get_state(self):
        row = self.data.iloc[self.current_idx]
        user_idx = torch.tensor([row['user_idx']], device='cpu', dtype=torch.long)
        item_idx = torch.tensor([row['item_idx']], device='cpu', dtype=torch.long)
        with torch.no_grad():
            u_emb = self.model.user_emb(user_idx).cpu().numpy().flatten()
            i_emb = self.model.item_emb(item_idx).cpu().numpy().flatten()
        return np.concatenate([u_emb, i_emb])  # استفاده از تمام ابعاد
    
    def _calculate_reward_components(self, selected_paths, true_rating, pred_rating, rmse):
        # دقت پیش‌بینی (اصلاح شده)
        accuracy = 1.0 / (1.0 + rmse)  # معکوس RMSE
        
        # پاداش اضافی برای پیش‌بینی دقیق
        if abs(pred_rating - true_rating) < 0.5:
            accuracy += 1.0
        
        # تنوع مسیرها (SID)
        sid = self._calculate_sid(selected_paths) if selected_paths else 0.0
        
        # ایمنی (محدوده ریتینگ)
        safety_penalty = 0
        if pred_rating < self.safety_constraints['min_rating']:
            safety_penalty = -0.5 * (self.safety_constraints['min_rating'] - pred_rating)
        elif pred_rating > self.safety_constraints['max_rating']:
            safety_penalty = -0.5 * (pred_rating - self.safety_constraints['max_rating'])
        
        # سادگی (تعداد مسیرهای انتخاب شده)
        simplicity = -0.05 * len(selected_paths)  # جریمه کمتر
        
        return {
            'accuracy': accuracy,
            'diversity': sid,
            'safety': safety_penalty,
            'simplicity': simplicity
        }
    
    def _calculate_sid(self, paths):
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
    
    def _generate_explanation(self, user_idx, item_idx, paths, true_rating, pred_rating):
        # تفسیر مبتنی بر مسیرها
        path_explanations = []
        for i, path in enumerate(paths):
            exp = []
            for j in range(len(path) - 1):
                from_node = path[j]
                to_node = path[j+1]
                rel = self.G.edges[from_node, to_node].get('relation_type', 'Rated_3')
                
                if 'UserSim' in rel:
                    exp.append(f"کاربر {from_node} ←→ کاربر {to_node}")
                elif 'ItemSim' in rel:
                    exp.append(f"فیلم {from_node} ←→ فیلم {to_node}")
                elif 'Rated' in rel:
                    rating = rel.split('_')[-1]
                    if G.nodes[from_node].get('node_type') == 'user':
                        exp.append(f"کاربر {from_node} → فیلم {to_node} ({rating})")
                    else:
                        exp.append(f"فیلم {from_node} → کاربر {to_node} ({rating})")
            path_explanations.append(" → ".join(exp))
        
        return {
            'user': user_idx.item(),
            'item': item_idx.item(),
            'true_rating': true_rating,
            'pred_rating': pred_rating,
            'paths': path_explanations,
            'num_paths': len(paths)
        }
    
    def save_explanations(self, filename):
        """ذخیره تاریخچه تفسیرها در فایل"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.explanation_history, f, ensure_ascii=False, indent=2)

# 8. آموزش و ارزیابی سیستم
def train_comper_with_rl():
    # مسیر پایه برای ذخیره‌سازی
    BASE_DIR = "C:/Users/jalal/Downloads/COMPER/ml-latest-small"
    
    # آماده‌سازی اولیه
    print("⚙️ آماده‌سازی مدل...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ دستگاه: {device}")
    
    # پارامترهای مدل
    num_users = len(user_ids)
    num_items = len(item_ids)
    relation_types = list(set(d['relation_type'] for _, _, d in G.edges(data=True)))
    relation2idx = {rel: idx for idx, rel in enumerate(relation_types)}
    feature_names = [f'Emb_{i}' for i in range(64)]  # متناسب با emb_dim=64
    
    # تقسیم داده‌ها
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"✂️ تقسیم داده: آموزش={len(train_data)}, تست={len(test_data)}")
    
    # ایجاد مدل COMPER - ارسال relation2idx به مدل
    model = AdvancedCOMPER(
        num_users, 
        num_items, 
        relation2idx,  # ارسال relation2idx
        emb_dim=64, 
        lstm_dim=64
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # پیش‌آموزش مدل پایه
    print("🔥 پیش‌آموزش مدل پایه...")
    train_data_small = train_data.sample(2000, random_state=42)  # افزایش نمونه
    
    for epoch in range(3):  # افزایش به 5 دوره
        epoch_loss = 0
        processed_count = 0
        progress_bar = tqdm(train_data_small.iterrows(), total=len(train_data_small), desc=f"دوره {epoch+1}/3")
        for _, row in progress_bar:
            # تبدیل به اندیس پیوسته
            user_idx = torch.tensor([row['user_idx']], device=device, dtype=torch.long)
            item_idx = torch.tensor([row['item_idx']], device=device, dtype=torch.long)
            true_rating = torch.tensor([row['rating']], dtype=torch.float, device=device)
            
            paths = extract_meta_paths(G, user_idx.item(), item_idx.item(), max_paths=5, max_length=4)
            
            # اگر مسیری وجود نداشت، از حلقه عبور کن
            if not paths:
                continue
                
            try:
                pred, _ = model(user_idx, item_idx, paths)
                loss = F.mse_loss(pred, true_rating)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                processed_count += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"خطا در آموزش: {e}")
        
        if processed_count > 0:
            avg_loss = epoch_loss / processed_count
        else:
            avg_loss = 0
            
        scheduler.step()
        print(f"دوره {epoch+1} - میانگین خطا: {avg_loss:.4f}")
    
    # ایجاد محیط یادگیری تقویتی
    print("🤖 ایجاد محیط RL...")
    env = AdvancedRecSysEnv(
        model, 
        train_data_small,
        G,
        relation2idx,
        feature_names
    )
    env = DummyVecEnv([lambda: env])
    
    # ایجاد عامل RL (PPO)
    rl_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=256,  # افزایش برای بهبود یادگیری
        batch_size=64,
        n_epochs=3,
        device=device
    )
    
    # آموزش عامل RL
    print("🚀 آموزش عامل RL...")
    rl_model.learn(total_timesteps=5000)  # افزایش مراحل آموزش
    
    # ذخیره مدل آموزش دیده
    rl_model_path = os.path.join(BASE_DIR, "comper_rl_model")
    rl_model.save(rl_model_path)
    print(f"💾 مدل RL ذخیره شد در: {rl_model_path}")
    
    # ارزیابی مدل
    print("🧪 ارزیابی مدل...")
    test_rewards = []
    all_true_ratings = []
    all_pred_ratings = []
    
    for i in tqdm(range(50), desc="ارزیابی"):  # افزایش اپیزودهای تست
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = rl_model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # جمع‌آوری داده‌ها برای محاسبه MAE و RMSE
            all_true_ratings.append(info[0]['true_rating'])
            all_pred_ratings.append(info[0]['pred_rating'])
        
        test_rewards.append(total_reward)
    
    # محاسبه معیارهای ارزیابی
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    
    # محاسبه MAE و RMSE
    mae = mean_absolute_error(all_true_ratings, all_pred_ratings)
    rmse = math.sqrt(mean_squared_error(all_true_ratings, all_pred_ratings))
    
    print(f"✅ میانگین پاداش تست: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"📊 معیارهای ارزیابی:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    
    # ذخیره تفسیرها
    explanation_path = os.path.join(BASE_DIR, "path_explanations.json")
    env.envs[0].save_explanations(explanation_path)
    print(f"💾 تفسیرهای مسیر ذخیره شدند در: {explanation_path}")
    
    # تولید گزارش نهایی
    generate_final_report(model, rl_model, test_rewards, all_true_ratings, all_pred_ratings, BASE_DIR)

def generate_final_report(model, rl_model, test_rewards, true_ratings, pred_ratings, base_dir):
    """تولید گزارش نهایی و نمودارها"""
    print("📊 تولید گزارش نهایی...")
    
    # محاسبه معیارهای ارزیابی
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    mae = mean_absolute_error(true_ratings, pred_ratings)
    rmse = math.sqrt(mean_squared_error(true_ratings, pred_ratings))
    
    # نمودار پاداش‌ها
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards)
    plt.title("پاداش‌های تست در طول ارزیابی", fontsize=14)
    plt.xlabel("اپیزود تست", fontsize=12)
    plt.ylabel("پاداش", fontsize=12)
    plt.grid(True)
    
    # محاسبه میانگین متحرک
    window_size = 5
    moving_avg = np.convolve(test_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(test_rewards)), moving_avg, 'r-', linewidth=2)
    
    # ذخیره نمودار
    plot_path = os.path.join(base_dir, "test_rewards.png")
    plt.savefig(plot_path)
    print(f"💾 نمودار پاداش‌ها ذخیره شد در: {plot_path}")
    
    # نمودار توزیع خطاها
    errors = [abs(t - p) for t, p in zip(true_ratings, pred_ratings)]
    plt.figure(figsize=(12, 8))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.title("توزیع خطاهای پیش‌بینی", fontsize=14)
    plt.xlabel("خطای مطلق (MAE)", fontsize=12)
    plt.ylabel("تعداد نمونه‌ها", fontsize=12)
    plt.grid(True)
    error_dist_path = os.path.join(base_dir, "error_distribution.png")
    plt.savefig(error_dist_path)
    print(f"💾 نمودار توزیع خطاها ذخیره شد در: {error_dist_path}")
    
    # نمودار مقایسه ریتینگ واقعی و پیش‌بینی شده
    plt.figure(figsize=(12, 8))
    plt.scatter(true_ratings, pred_ratings, alpha=0.3)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.title("مقایسه ریتینگ واقعی و پیش‌بینی شده", fontsize=14)
    plt.xlabel("ریتینگ واقعی", fontsize=12)
    plt.ylabel("ریتینگ پیش‌بینی شده", fontsize=12)
    plt.grid(True)
    comparison_path = os.path.join(base_dir, "rating_comparison.png")
    plt.savefig(comparison_path)
    print(f"💾 نمودار مقایسه ریتینگ‌ها ذخیره شد در: {comparison_path}")
    
    # ذخیره وزن‌های مدل
    weights_path = os.path.join(base_dir, "comper_model_weights.pth")
    torch.save(model.state_dict(), weights_path)
    
    # گزارش متنی
    report = f"""
    ================== گزارش نهایی سیستم COMPER ==================
    عملکرد سیستم:
      - میانگین پاداش تست: {avg_reward:.4f} ± {std_reward:.4f}
      - حداکثر پاداش: {np.max(test_rewards):.4f}
      - حداقل پاداش: {np.min(test_rewards):.4f}
      - MAE (میانگین خطای مطلق): {mae:.4f}
      - RMSE (ریشه میانگین مربعات خطا): {rmse:.4f}
    
    اطلاعات مدل:
      - معماری: COMPER پیشرفته با لایه‌های LSTM و توجه
      - ابعاد جاسازی: 64
      - پارامترهای مدل: {sum(p.numel() for p in model.parameters())}
      - نوع RL: PPO با سیاست MLP
    
    خروجی‌ها:
      - مدل RL: {os.path.join(base_dir, "comper_rl_model.zip")}
      - وزن‌های COMPER: {weights_path}
      - تفسیرهای مسیر: {os.path.join(base_dir, "path_explanations.json")}
      - نمودار پاداش‌ها: {plot_path}
      - نمودار توزیع خطاها: {error_dist_path}
      - نمودار مقایسه ریتینگ‌ها: {comparison_path}
    =============================================================
    """
    print(report)
    
    # ذخیره گزارش متنی
    report_path = os.path.join(base_dir, "final_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    
    # ذخیره نتایج ارزیابی در فایل CSV
    eval_results = pd.DataFrame({
        'true_rating': true_ratings,
        'pred_rating': pred_ratings,
        'error': [abs(t - p) for t, p in zip(true_ratings, pred_ratings)]
    })
    eval_path = os.path.join(base_dir, "evaluation_results.csv")
    eval_results.to_csv(eval_path, index=False)
    print(f"💾 نتایج ارزیابی ذخیره شد در: {eval_path}")
    
    print(f"💾 گزارش نهایی ذخیره شد در: {report_path}")

# اجرای سیستم
if __name__ == "__main__":
    train_comper_with_rl()
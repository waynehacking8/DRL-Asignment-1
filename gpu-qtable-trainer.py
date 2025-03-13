import torch
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
from collections import deque

# 匯入測試環境
from simple_custom_taxi_env import SimpleTaxiEnv

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GPUSuperiorQTableTrainer:
    """GPU加速的進階強化學習訓練器，優化成功率並減少文件雜亂"""
    
    def __init__(self):
        self.q_table = {}
        self.load_existing_qtable()
        
        # 追蹤訓練進展
        self.rewards_history = []
        self.success_history = []
        
        # 優化的訓練參數
        self.alpha_start = 0.15        # 初始學習率 (略微降低以增加穩定性)
        self.alpha_end = 0.01          # 最終學習率
        self.gamma = 0.995             # 折扣因子 (提高以更重視長期獎勵)
        self.epsilon_start = 1.0       # 初始探索率
        self.epsilon_end = 0.01        # 最終探索率
        
        # 經驗回放增強
        self.memory = []               # 經驗回放記憶
        self.memory_size = 20000       # 記憶容量 (增加)
        self.batch_size = 256          # 批量大小 (增加以充分利用GPU)
        self.min_memory_size = 1000    # 開始學習的最小記憶量 (增加)
        self.priority_alpha = 0.6      # 優先級誇大因子
        
        # 計數器和追蹤器
        self.total_steps = 0
        self.episode_count = 0
        self.best_success_rate = 0
        self.best_success_count = 0
        self.success_window = deque(maxlen=100)  # 用於計算成功率的窗口
        
        # 保存設置
        self.save_best_only = True     # 只保存最佳模型，減少文件雜亂
    
    def load_existing_qtable(self):
        """載入現有的Q-table並轉換為GPU張量"""
        if os.path.exists("q_table.pkl"):
            try:
                with open("q_table.pkl", "rb") as f:
                    loaded_q_table = pickle.load(f)
                
                # 轉換為GPU張量
                self.q_table = {
                    state: torch.tensor(values, device=device, dtype=torch.float32)
                    for state, values in loaded_q_table.items()
                }
                print(f"Loaded existing Q-table with {len(self.q_table)} states")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                self.q_table = {}
        else:
            print("No existing Q-table found. Starting fresh.")
            self.q_table = {}
    
    def save_qtable(self, is_best=False):
        """保存Q-table，先將GPU張量轉換回NumPy數組"""
        # 將GPU張量轉換回NumPy
        cpu_q_table = {
            state: values.cpu().numpy() if isinstance(values, torch.Tensor) else values
            for state, values in self.q_table.items()
        }
        
        # 保存轉換後的Q-table
        with open("q_table.pkl", "wb") as f:
            pickle.dump(cpu_q_table, f)
        
        if is_best and self.save_best_only:
            with open(f"q_table_best.pkl", "wb") as f:
                pickle.dump(cpu_q_table, f)
            print(f"Best model saved with {len(self.q_table)} states (success rate: {self.best_success_rate:.1f}%)")
    
    def add_to_memory(self, state, action, reward, next_state, done, priority=None):
        """添加經驗到記憶，支持優先級"""
        if priority is None:
            # 計算簡單優先級：獎勵絕對值 + 完成標誌
            priority = abs(reward) + (10 if done else 0)
        
        if len(self.memory) >= self.memory_size:
            # 如果記憶已滿，移除優先級最低的項目
            if len(self.memory) > 0:
                min_priority_idx = np.argmin([mem[5] for mem in self.memory])
                self.memory.pop(min_priority_idx)
        
        self.memory.append((state, action, reward, next_state, done, priority))
    
    def sample_from_memory(self):
        """從記憶中進行優先採樣"""
        if len(self.memory) < self.min_memory_size:
            return None
        
        # 計算優先級
        priorities = np.array([mem[5] for mem in self.memory]) ** self.priority_alpha
        probabilities = priorities / np.sum(priorities)
        
        # 優先採樣
        indices = np.random.choice(len(self.memory), size=min(self.batch_size, len(self.memory)), 
                                   p=probabilities, replace=False)
        
        return [self.memory[idx] for idx in indices]
    
    def train_from_memory(self):
        """使用GPU加速從記憶中學習，使用批量處理"""
        sampled_batch = self.sample_from_memory()
        if not sampled_batch:
            return 0
        
        # 計算當前學習率
        progress = min(1.0, self.episode_count / 5000)
        alpha = self.alpha_start - progress * (self.alpha_start - self.alpha_end)
        
        # 準備批量數據
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        memory_indices = []
        
        for i, (state, action, reward, next_state, done, _) in enumerate(sampled_batch):
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            memory_indices.append(i)
            
            # 確保狀態存在於Q-table
            if state not in self.q_table:
                self.q_table[state] = torch.zeros(6, device=device, dtype=torch.float32)
            
            if next_state not in self.q_table and not done:
                self.q_table[next_state] = torch.zeros(6, device=device, dtype=torch.float32)
        
        # 將獎勵和完成標誌轉換為張量
        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        done_tensor = torch.tensor(dones, device=device, dtype=torch.float32)
        action_tensor = torch.tensor(actions, device=device, dtype=torch.long)
        
        # 獲取當前Q值的批量處理
        current_q_values = torch.stack([self.q_table[state][action] for state, action in zip(states, actions)])
        
        # 計算目標Q值
        target_q_values = torch.zeros_like(current_q_values, device=device)
        
        # 對於非終止狀態，計算 reward + gamma * max_future_q
        non_final_mask = ~done_tensor.bool()
        non_final_next_states = [next_states[i] for i in range(len(next_states)) if not dones[i]]
        
        if non_final_next_states:  # 確保有非終止狀態
            max_future_q_values = torch.stack([self.q_table[state].max() for state in non_final_next_states])
            target_q_values[non_final_mask] = reward_tensor[non_final_mask] + self.gamma * max_future_q_values
        
        # 對於終止狀態，僅使用獎勵
        target_q_values[done_tensor.bool()] = reward_tensor[done_tensor.bool()]
        
        # 計算損失和TD誤差
        td_errors = torch.abs(target_q_values - current_q_values)
        loss = torch.mean(td_errors ** 2)
        
        # 更新Q值
        new_q_values = current_q_values + alpha * (target_q_values - current_q_values)
        
        # 更新Q-table
        for i in range(len(states)):
            state, action = states[i], actions[i]
            self.q_table[state][action] = new_q_values[i]
        
        # 更新記憶中的優先級
        for i, (state, action, reward, next_state, done, _) in enumerate(sampled_batch):
            # 使用新計算的TD誤差作為優先級
            idx = memory_indices[i]
            td_error = td_errors[i].item()
            self.memory[i] = (state, action, reward, next_state, done, td_error + 0.01)  # 加0.01確保非零優先級
        
        return loss.item()
    
    def get_smart_action(self, state, epsilon):
        """獲取智能動作，改進的探索策略，支持GPU"""
        # 解析狀態
        if len(state) >= 11:
            taxi_row, taxi_col, pass_idx, dest_idx, in_taxi = state[0:5]
            rel_row, rel_col = state[5:7]
            obstacles = state[7:11]
            
            # 優先考慮pickup和dropoff
            if in_taxi == 0 and rel_row == 0 and rel_col == 0:
                return 4  # Pickup
            if in_taxi == 1 and rel_row == 0 and rel_col == 0:
                return 5  # Dropoff
            
            # 檢查Q-table
            if state in self.q_table and np.random.random() > epsilon:
                # 選擇最高Q值的動作，但考慮障礙物
                q_values = self.q_table[state].clone().cpu().numpy() if isinstance(self.q_table[state], torch.Tensor) else self.q_table[state].copy()
                
                # 動作遮罩 - 對有障礙物的方向賦予極低的Q值
                for i, obstacle in enumerate(obstacles):
                    if obstacle == 1 and i < 4:  # 只考慮移動動作
                        q_values[i] = -np.inf
                
                # 如果有可行動作，選擇最佳的
                if np.max(q_values) > -np.inf:
                    return int(np.argmax(q_values))
            
            # 智能探索 - 考慮目標位置和障礙物
            valid_actions = []
            action_weights = []
            
            # 評估每個移動方向
            # 北 (1)
            if obstacles[0] == 0:  # 沒有障礙物
                valid_actions.append(1)
                weight = 1.0
                if rel_row < 0:  # 目標在北方
                    weight += abs(rel_row) * 2
                action_weights.append(weight)
            
            # 南 (0)
            if obstacles[1] == 0:
                valid_actions.append(0)
                weight = 1.0
                if rel_row > 0:  # 目標在南方
                    weight += abs(rel_row) * 2
                action_weights.append(weight)
            
            # 東 (2)
            if obstacles[2] == 0:
                valid_actions.append(2)
                weight = 1.0
                if rel_col > 0:  # 目標在東方
                    weight += abs(rel_col) * 2
                action_weights.append(weight)
            
            # 西 (3)
            if obstacles[3] == 0:
                valid_actions.append(3)
                weight = 1.0
                if rel_col < 0:  # 目標在西方
                    weight += abs(rel_col) * 2
                action_weights.append(weight)
            
            # 如果有可行動作，根據權重選擇一個
            if valid_actions:
                # 將權重轉換為概率
                total_weight = sum(action_weights)
                probs = [w/total_weight for w in action_weights]
                return np.random.choice(valid_actions, p=probs)
            
            # 如果沒有可行的移動方向，嘗試pickup/dropoff
            return 4 if in_taxi == 0 else 5
        
        # 如果狀態格式不正確，返回隨機動作
        return np.random.randint(0, 6)
    
    def train(self, episodes=7000, max_steps=200):
        """高級訓練函數，優化成功率，使用GPU加速"""
        print(f"Starting GPU-accelerated training for {episodes} episodes...")
        start_time = time.time()
        
        # 初始化進度條
        pbar = tqdm(range(1, episodes + 1))
        
        # 訓練前的預填充記憶
        print("Pre-filling memory with initial experiences...")
        self.prefill_memory(1000)
        
        for episode in pbar:
            self.episode_count = episode
            
            # 計算當前探索率 (使用更平滑的衰減)
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-5 * episode / episodes)
            
            # 隨機選擇環境配置，但更偏向於小網格（更容易成功）
            grid_sizes = [5, 5, 6, 6, 7, 7, 8, 9, 10]  # 偏向小網格
            grid_size = np.random.choice(grid_sizes)
            obstacle_density = np.random.choice([0.1, 0.1, 0.15, 0.15, 0.2])  # 偏向低障礙物密度
            
            # 創建環境
            env = SimpleTaxiEnv(grid_size=grid_size, obstacle_density=obstacle_density)
            
            # 重置環境
            state = env.reset()
            state = tuple(state)
            
            # 回合變數
            done = False
            total_reward = 0
            step_count = 0
            
            # 對抗循環的變數
            pos_history = deque(maxlen=10)
            in_cycle = False
            
            # 回合循環
            while not done and step_count < max_steps:
                # 檢測循環
                if len(state) >= 5:
                    taxi_row, taxi_col, _, _, in_taxi = state[0:5]
                    current_pos = (taxi_row, taxi_col, in_taxi)
                    
                    if current_pos in pos_history:
                        in_cycle = True
                    
                    pos_history.append(current_pos)
                
                # 選擇動作
                if in_cycle:
                    # 檢測到循環，強制探索
                    action = self.get_smart_action(state, epsilon=1.0)
                    in_cycle = False  # 重置循環標誌
                else:
                    # 正常動作選擇
                    action = self.get_smart_action(state, epsilon)
                
                # 執行動作
                next_state, reward, done, _ = env.step(action)
                next_state = tuple(next_state)
                total_reward += reward
                
                # 添加到記憶
                # 為重要的經驗賦予更高優先級
                priority = abs(reward)
                if reward > 0:  # 正面獎勵（pickup或成功dropoff）
                    priority += 10
                if done and reward > 0:  # 成功完成
                    priority += 50
                
                self.add_to_memory(state, action, reward, next_state, done, priority)
                
                # 移動到下一個狀態
                state = next_state
                step_count += 1
                self.total_steps += 1
                
                # 定期從記憶中學習
                if self.total_steps % 4 == 0:  # 增加學習頻率以利用GPU
                    self.train_from_memory()
            
            # 記錄成功情況
            success = total_reward > 35  # 獎勵>35通常表示成功
            self.success_window.append(success)
            
            # 更新獎勵歷史
            self.rewards_history.append(total_reward)
            self.success_history.append(success)
            
            # 計算最近100回合的成功率
            recent_success_rate = sum(self.success_window) * 100 / len(self.success_window)
            
            # 檢查是否是新的最佳成功率
            if recent_success_rate > self.best_success_rate and len(self.success_window) >= 50:
                self.best_success_rate = recent_success_rate
                self.save_qtable(is_best=True)
            
            # 每1000回合保存一次主要Q-table
            if episode % 1000 == 0:
                self.save_qtable(is_best=False)
                print(f"Regular checkpoint: Q-table with {len(self.q_table)} states saved")
            
            # 更新進度條顯示
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0
                
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.2f}',
                    'success': f'{recent_success_rate:.1f}%',
                    'epsilon': f'{epsilon:.3f}',
                    'states': len(self.q_table)
                })
        
        # 保存最終Q-table
        self.save_qtable()
        
        # 計算訓練時間
        training_time = time.time() - start_time
        minutes, seconds = divmod(training_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        print(f"Training complete in {int(hours)}h {int(minutes)}m {int(seconds)}s!")
        print(f"Final Q-table has {len(self.q_table)} states")
        print(f"Best success rate: {self.best_success_rate:.1f}%")
        
        # 評估最終表現
        if len(self.rewards_history) >= 1000:
            final_rewards = self.rewards_history[-1000:]
            final_successes = self.success_history[-1000:]
        else:
            final_rewards = self.rewards_history
            final_successes = self.success_history
            
        final_avg_reward = np.mean(final_rewards)
        final_success_rate = np.mean(final_successes) * 100
        
        print(f"Final average reward: {final_avg_reward:.2f}")
        print(f"Final success rate: {final_success_rate:.1f}%")
        
        # 運行最終評估
        self.evaluate()
    
    def prefill_memory(self, num_experiences=1000):
        """預填充經驗回放記憶"""
        collected = 0
        pbar = tqdm(total=num_experiences, desc="Collecting initial experiences")
        
        while collected < num_experiences:
            # 建立一個簡單環境
            grid_size = np.random.choice([5, 6, 7])
            obstacle_density = np.random.choice([0.1, 0.15])
            env = SimpleTaxiEnv(grid_size=grid_size, obstacle_density=obstacle_density)
            
            state = env.reset()
            state = tuple(state)
            done = False
            steps = 0
            
            while not done and steps < 100 and collected < num_experiences:
                # 使用純探索策略
                action = self.get_smart_action(state, epsilon=1.0)
                
                next_state, reward, done, _ = env.step(action)
                next_state = tuple(next_state)
                
                # 添加經驗
                priority = abs(reward) + (10 if done and reward > 0 else 0)
                self.add_to_memory(state, action, reward, next_state, done, priority)
                
                state = next_state
                steps += 1
                collected += 1
                pbar.update(1)
        
        pbar.close()
    
    def evaluate(self, num_episodes=100):
        """評估智能體表現"""
        print(f"\nEvaluating agent over {num_episodes} episodes...")
        
        # 轉換為CPU模式進行評估
        cpu_q_table = {
            state: values.cpu().numpy() if isinstance(values, torch.Tensor) else values
            for state, values in self.q_table.items()
        }
        
        # 創建臨時文件以供student_agent.py使用
        with open("q_table_eval.pkl", "wb") as f:
            pickle.dump(cpu_q_table, f)
        
        original_q_table_file = "q_table.pkl"
        backup_exists = False
        
        # 備份原始q_table.pkl
        if os.path.exists(original_q_table_file):
            os.rename(original_q_table_file, "q_table.pkl.bak")
            backup_exists = True
        
        # 使用評估q_table
        os.rename("q_table_eval.pkl", original_q_table_file)
        
        try:
            from student_agent import get_action  # 使用student_agent.py中的get_action函數
            
            total_rewards = []
            success_count = 0
            
            # 建立進度條
            pbar = tqdm(range(num_episodes))
            
            for episode in pbar:
                # 隨機選擇網格大小和障礙物密度
                grid_size = np.random.choice(range(5, 11))
                obstacle_density = np.random.choice([0.1, 0.15, 0.2])
                
                # 創建環境
                env = SimpleTaxiEnv(grid_size=grid_size, obstacle_density=obstacle_density)
                
                # 重置環境
                state = env.reset()
                
                # 回合變數
                done = False
                episode_reward = 0
                steps = 0
                
                # 回合循環
                while not done and steps < 200:
                    # 獲取動作
                    action = get_action(state)
                    
                    # 執行動作
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    # 更新狀態
                    state = next_state
                    steps += 1
                
                # 追蹤表現
                total_rewards.append(episode_reward)
                if episode_reward > 35:  # 通常表示成功
                    success_count += 1
                
                # 更新進度條
                pbar.set_postfix({
                    'reward': f'{episode_reward:.2f}',
                    'success_rate': f'{(success_count/(episode+1))*100:.1f}%'
                })
            
            # 計算最終表現
            avg_reward = np.mean(total_rewards)
            success_rate = (success_count / num_episodes) * 100
            
            print(f"\nEvaluation Results:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Success Rate: {success_rate:.1f}%")
            
            return avg_reward, success_rate
            
        finally:
            # 恢復原始q_table.pkl
            os.remove(original_q_table_file)
            if backup_exists:
                os.rename("q_table.pkl.bak", original_q_table_file)

def main():
    """訓練和評估高精度出租車智能體，使用GPU加速"""
    trainer = GPUSuperiorQTableTrainer()
    
    # 訓練參數
    episodes = 7000  # 訓練回合數
    
    # 訓練智能體
    trainer.train(episodes=episodes)

if __name__ == "__main__":
    main()
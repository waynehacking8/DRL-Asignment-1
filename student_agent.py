import numpy as np
import pickle
import os
import torch
from collections import deque

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPUSuperTaxiAgent:
    """
    GPU加速的超高精度出租車智能體，專為達到80%+成功率而設計。
    結合Q-learning與超級路徑規劃算法，使用GPU加速。
    """
    
    def __init__(self):
        # 載入Q-table
        self.q_table = {}
        try:
            # 優先載入最佳模型（如果存在）
            if os.path.exists("q_table_best.pkl"):
                with open("q_table_best.pkl", "rb") as f:
                    loaded_q_table = pickle.load(f)
                print(f"Loaded best Q-table with {len(loaded_q_table)} states")
            elif os.path.exists("q_table.pkl"):
                with open("q_table.pkl", "rb") as f:
                    loaded_q_table = pickle.load(f)
                print(f"Loaded standard Q-table with {len(loaded_q_table)} states")
            else:
                loaded_q_table = {}
                print("No Q-table found")
            
            # 將Q-table轉換為GPU張量（如果可用）
            self.q_table = {
                state: torch.tensor(values, device=device, dtype=torch.float32) if isinstance(values, np.ndarray) else values
                for state, values in loaded_q_table.items()
            }
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            self.q_table = {}
        
        # 進階循環檢測與路徑規劃
        self.position_history = deque(maxlen=15)    # 位置歷史
        self.action_history = deque(maxlen=10)      # 動作歷史
        self.observed_rewards = deque(maxlen=10)    # 獎勵歷史
        self.stuck_counter = 0                      # 卡住計數器
        self.episode_step = 0                       # 回合步數
        
        # 重置標誌
        self.has_seen_first_state = False
        self.first_state = None
        
        # 行為參數
        self.exploration_rate = 0.05                # 隨機探索率
        self.detect_cycles_threshold = 5            # 循環檢測閾值
        self.path_planning_depth = 4                # 路徑規劃深度
        
        # 性能追蹤
        self.total_rewards = 0
        self.pickup_attempts = 0
        self.successful_pickups = 0
        self.dropoff_attempts = 0
        self.successful_dropoffs = 0
        
        # GPU性能優化
        self.use_gpu = torch.cuda.is_available()
        self.batch_observations = []                # 批量處理的觀察
        self.batch_size = 32                        # 批量大小
    
    def clear_history(self):
        """清除歷史記錄，用於新回合"""
        self.position_history.clear()
        self.action_history.clear()
        self.observed_rewards.clear()
        self.stuck_counter = 0
        self.episode_step = 0
        self.total_rewards = 0
        self.has_seen_first_state = False
        self.first_state = None
        self.batch_observations.clear()
    
    def detect_cycle(self, position):
        """超級循環檢測"""
        # 添加位置到歷史
        self.position_history.append(position)
        
        # 檢測方法1: 頻繁訪問相同位置
        if len(self.position_history) >= self.detect_cycles_threshold:
            recent_positions = list(self.position_history)[-self.detect_cycles_threshold:]
            for pos in set(recent_positions):
                if recent_positions.count(pos) >= 3:
                    return True
        
        # 檢測方法2: 動作循環模式
        if len(self.action_history) >= 6:
            actions = list(self.action_history)
            # 檢測2-動作循環 (如 北-南-北-南-北-南)
            if (actions[-1] == actions[-3] == actions[-5] and 
                actions[-2] == actions[-4] == actions[-6] and
                actions[-1] != actions[-2]):
                return True
            
            # 檢測3-動作循環
            if len(self.action_history) >= 9:
                if (actions[-1] == actions[-4] == actions[-7] and 
                    actions[-2] == actions[-5] == actions[-8] and 
                    actions[-3] == actions[-6] == actions[-9]):
                    return True
        
        # 檢測方法3: 獎勵停滯
        if len(self.observed_rewards) >= 8:
            # 如果連續多個回合的獎勵為負且相同，可能陷入循環
            if all(r == self.observed_rewards[0] and r <= 0 for r in self.observed_rewards):
                return True
        
        return False
    
    def update_history(self, action, reward):
        """更新智能體的歷史"""
        self.action_history.append(action)
        self.observed_rewards.append(reward)
        self.episode_step += 1
        self.total_rewards += reward
        
        # 追蹤pickup和dropoff成功率
        if action == 4:  # Pickup
            self.pickup_attempts += 1
            if reward > 0:
                self.successful_pickups += 1
        elif action == 5:  # Dropoff
            self.dropoff_attempts += 1
            if reward > 0:
                self.successful_dropoffs += 1
    
    def super_path_planning(self, state, obstacles, in_cycle=False):
        """
        GPU加速的超級路徑規劃算法
        考慮曼哈頓距離、障礙物、循環檢測、Q值及獎勵歷史
        """
        taxi_row, taxi_col, pass_idx, dest_idx, in_taxi, rel_row, rel_col = state[:7]
        
        # 優先處理pickup/dropoff
        if in_taxi == 0 and rel_row == 0 and rel_col == 0:
            return 4  # Pickup
        if in_taxi == 1 and rel_row == 0 and rel_col == 0:
            return 5  # Dropoff
        
        # 檢查Q-table中的值（如果存在且不在循環中）
        if state in self.q_table and not in_cycle and np.random.random() > self.exploration_rate:
            # 獲取Q值
            if self.use_gpu and isinstance(self.q_table[state], torch.Tensor):
                q_values = self.q_table[state].clone().cpu().numpy()
            else:
                q_values = np.array(self.q_table[state])
            
            # 將障礙物方向的Q值設為負無窮
            for i, obstacle in enumerate(obstacles[:4]):
                if obstacle == 1:
                    q_values[i] = -np.inf
            
            # 如果乘客不在正確位置，禁用pickup
            if in_taxi == 0 and (rel_row != 0 or rel_col != 0):
                q_values[4] = -np.inf
                
            # 如果乘客不在車上或不在目的地，禁用dropoff
            if in_taxi == 0 or (in_taxi == 1 and (rel_row != 0 or rel_col != 0)):
                q_values[5] = -np.inf
            
            # 如果有有效的Q值，使用它
            if np.max(q_values) > -np.inf:
                return int(np.argmax(q_values))
        
        # 高級路徑規劃 - 為每個動作計算多層次分數
        action_scores = np.zeros(4)  # [南, 北, 東, 西]
        
        # 層次1: 基於相對位置的分數
        if rel_row > 0:  # 目標在南方
            action_scores[0] += abs(rel_row) * 3  # 南
        elif rel_row < 0:  # 目標在北方
            action_scores[1] += abs(rel_row) * 3  # 北
            
        if rel_col > 0:  # 目標在東方
            action_scores[2] += abs(rel_col) * 3  # 東
        elif rel_col < 0:  # 目標在西方
            action_scores[3] += abs(rel_col) * 3  # 西
        
        # 層次2: 考慮曼哈頓距離
        manhattan_dist = abs(rel_row) + abs(rel_col)
        # 如果距離較遠，增加在主要方向上移動的權重
        if manhattan_dist > 3:
            if abs(rel_row) > abs(rel_col):
                # 行距離更遠，優先垂直移動
                if rel_row > 0:
                    action_scores[0] *= 1.5  # 增加南移權重
                else:
                    action_scores[1] *= 1.5  # 增加北移權重
            else:
                # 列距離更遠，優先水平移動
                if rel_col > 0:
                    action_scores[2] *= 1.5  # 增加東移權重
                else:
                    action_scores[3] *= 1.5  # 增加西移權重
        
        # 層次3: 障礙物處理
        for i, obstacle in enumerate(obstacles[:4]):
            if obstacle == 1:
                action_scores[i] = -np.inf  # 無法移動到障礙物
        
        # 層次4: 考慮過去動作，避免重複
        if self.action_history:
            last_action = self.action_history[-1]
            if 0 <= last_action < 4:  # 只考慮移動動作
                # 降低最近使用動作的分數
                action_scores[last_action] *= 0.8
                
                # 降低與上一個動作相反方向的分數，避免來回移動
                opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}  # 南北東西的相反方向
                action_scores[opposite_actions[last_action]] *= 0.7
        
        # 層次5: 如果檢測到循環，顯著改變行為
        if in_cycle:
            # 降低最近常用動作的分數
            for action in self.action_history:
                if 0 <= action < 4:
                    action_scores[action] *= 0.5
            
            # 增加隨機性
            action_scores = action_scores + np.random.uniform(0, 0.5, 4)
        
        # 選擇最高分數的動作
        if np.max(action_scores) > -np.inf:
            return int(np.argmax(action_scores))
        
        # 如果所有方向都被阻擋，隨機選擇一個可能的動作
        possible_actions = []
        for i, obstacle in enumerate(obstacles[:4]):
            if obstacle == 0:  # 無障礙物
                possible_actions.append(i)
        
        if possible_actions:
            return np.random.choice(possible_actions)
        
        # 如果真的無路可走，返回pickup/dropoff (極少發生)
        return 4 if in_taxi == 0 else 5
    
    def detect_new_episode(self, state):
        """檢測是否開始了新回合"""
        if not self.has_seen_first_state:
            self.first_state = state[:2]  # 記錄第一個狀態的位置
            self.has_seen_first_state = True
            return False
        
        # 如果出租車位置與第一個狀態相距太遠，可能是新回合
        current_pos = state[:2]
        if self.first_state and manhattan_distance(current_pos, self.first_state) > 3:
            return True
        
        return False
    
    def get_action(self, obs):
        """
        獲取最優動作，支持GPU加速

        Args:
            obs: 環境觀察

        Returns:
            int: 選定的動作 (0-5)
        """
        # 確保觀察是元組形式
        state = tuple(obs) if isinstance(obs, (list, np.ndarray)) else obs
        
        # 檢測是否是新回合
        if self.detect_new_episode(state):
            self.clear_history()
        
        # 提取狀態信息
        if len(state) >= 11:
            taxi_row, taxi_col, pass_idx, dest_idx, in_taxi = state[0:5]
            rel_row, rel_col = state[5:7]
            obstacles = state[7:11]  # 北、南、東、西方向的障礙物
        else:
            # 狀態格式不正確，使用隨機動作
            return np.random.randint(0, 6)
        
        # 當前位置
        current_position = (taxi_row, taxi_col, in_taxi)
        
        # 直接處理pickup/dropoff情況
        if in_taxi == 0 and rel_row == 0 and rel_col == 0:
            action = 4  # pickup
            self.update_history(action, 10)  # 假設成功的pickup獎勵為10
            return action
            
        if in_taxi == 1 and rel_row == 0 and rel_col == 0:
            action = 5  # dropoff
            self.update_history(action, 50)  # 假設成功的dropoff獎勵為50
            return action
        
        # 檢測循環
        in_cycle = self.detect_cycle(current_position)
        if in_cycle:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # 選擇動作
        action = self.super_path_planning(
            state, 
            obstacles,
            in_cycle=(in_cycle or self.stuck_counter > 3)
        )
        
        # 更新歷史 (假設移動獎勵為-0.1)
        self.update_history(action, -0.1)
        
        return action


def manhattan_distance(pos1, pos2):
    """計算曼哈頓距離"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# 創建全局智能體實例
agent = GPUSuperTaxiAgent()

def get_action(obs):
    """評估系統調用的接口函數"""
    return agent.get_action(obs)
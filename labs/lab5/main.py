import copy
import csv
import os
import random
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPLCONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib_cache")
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)
os.environ.setdefault("XDG_CACHE_HOME", MPLCONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Воспроизводимость
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PROTOCOL_DIR, exist_ok=True)

TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")
TRAINING_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.png")
POLICY_PATH = os.path.join(ARTIFACTS_DIR, "policy.pt")
TRAJECTORY_PATH = os.path.join(ARTIFACTS_DIR, "landing_trajectory.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Своя среда: мягкая посадка
# =========================
class SoftLandingEnv(gym.Env):
    """Одномерная посадка аппарата под действием гравитации.

    Состояние: [высота, вертикальная скорость, остаток топлива] (нормированные).
    Действия: 0 - двигатель выключен, 1 - малая тяга, 2 - полная тяга.
    Цель: коснуться поверхности (h <= 0) с малой скоростью (|v| < V_SAFE).
    """

    metadata = {"render_modes": []}

    DT = 0.1
    GRAVITY = 1.0
    THRUST = (0.0, 0.8, 2.0)          # ускорение тяги для действий 0/1/2
    FUEL_COST = (0.0, 0.15, 0.4)      # расход топлива за шаг для действий 0/1/2
    FUEL_START = 40.0
    H_START = (6.0, 9.0)              # диапазон стартовой высоты
    V_START = (-1.0, 0.0)             # диапазон стартовой скорости (вниз)
    H_SCALE = 9.0
    V_SCALE = 5.0
    V_SAFE = 0.6                      # порог мягкой посадки
    MAX_STEPS = 300

    def __init__(self):
        super().__init__()
        # высота, скорость, топливо (нормированные)
        high = np.array([1.5, 1.5, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.h = 0.0
        self.v = 0.0
        self.fuel = 0.0
        self.steps = 0

    def _obs(self):
        return np.array(
            [self.h / self.H_SCALE, self.v / self.V_SCALE, self.fuel / self.FUEL_START],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.h = float(self.np_random.uniform(*self.H_START))
        self.v = float(self.np_random.uniform(*self.V_START))
        self.fuel = self.FUEL_START
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)

        # тяга доступна только при наличии топлива
        if self.fuel > 0.0:
            thrust = self.THRUST[action]
            cost = self.FUEL_COST[action]
        else:
            thrust = 0.0
            cost = 0.0

        self.fuel = max(0.0, self.fuel - cost)
        self.v += (thrust - self.GRAVITY) * self.DT
        self.h += self.v * self.DT
        self.steps += 1

        terminated = False
        truncated = False

        # Небольшой штраф за шаг (не зависать) и за расход топлива.
        reward = -0.05 - 0.02 * cost

        if self.h <= 0.0:
            self.h = 0.0
            terminated = True
            impact = abs(self.v)
            # Гладкая награда: чем мягче касание, тем больше очков (монотонно).
            reward += 100.0 - 120.0 * impact
            reward += 20.0 * (self.fuel / self.FUEL_START)  # бонус за экономию топлива
            if impact < self.V_SAFE:
                reward += 50.0  # бонус за мягкую посадку
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            reward += -50.0  # завис / не сел в лимит времени

        return self._obs(), float(reward), terminated, truncated, {}


# =========================
# Q-сеть
# =========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Гиперпараметры
# =========================
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 64
MEMORY_SIZE = 20000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.993
TAU = 0.005           # коэффициент мягкого обновления целевой сети (Polyak)
EPISODES = 600
EVAL_EVERY = 20       # как часто оценивать политику и сохранять лучший чекпойнт


# =========================
# Агент Double DQN
# =========================
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPS_START
        self.total_steps = 0

    def select_action(self, state, greedy=False):
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN: действие выбирает policy_net, оценивает target_net
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Мягкое (Polyak) обновление целевой сети для стабильности.
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(),
                                           self.policy_net.parameters()):
                target_param.mul_(1.0 - TAU).add_(TAU * param)

        self.total_steps += 1
        return float(loss.item())


# =========================
# Обучение
# =========================
def train(env, agent):
    rewards_history = []
    log_rows = []
    best_state = copy.deepcopy(agent.policy_net.state_dict())
    best_score = (-1.0, -float("inf"))  # (доля мягких посадок, -средний удар)
    best_episode = 0
    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        soft = False
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward
            if done:
                soft = terminated and abs(env.v) < env.V_SAFE
                break

        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)
        rewards_history.append(total_reward)
        log_rows.append((episode, total_reward, agent.epsilon, int(soft)))

        if (episode + 1) % EVAL_EVERY == 0:
            m = evaluate(env, agent, episodes=20)
            score = (m["success_rate"], -m["avg_impact"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.policy_net.state_dict())
                best_episode = episode + 1
            window = rewards_history[-EVAL_EVERY:]
            print(
                f"Эпизод {episode + 1:>4d}/{EPISODES} | "
                f"avg{EVAL_EVERY}={np.mean(window):8.2f} | "
                f"eps={agent.epsilon:.3f} | "
                f"eval: мягких={m['success_rate']:.0%}, |v|={m['avg_impact']:.3f}"
            )

    print(f"\nЛучшая политика: эпизод {best_episode}, "
          f"мягких={best_score[0]:.0%}, |v|={-best_score[1]:.3f}")
    agent.policy_net.load_state_dict(best_state)
    agent.target_net.load_state_dict(best_state)
    return rewards_history, log_rows, best_episode


# =========================
# Оценка обученного агента
# =========================
def evaluate(env, agent, episodes=50):
    softs = 0
    impacts = []
    fuels = []
    for i in range(episodes):
        state, _ = env.reset(seed=10_000 + i)
        while True:
            action = agent.select_action(state, greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if terminated and abs(env.v) < env.V_SAFE:
            softs += 1
        impacts.append(abs(env.v) if terminated else float("nan"))
        fuels.append(env.fuel)
    return {
        "episodes": episodes,
        "success_rate": softs / episodes,
        "avg_impact": float(np.nanmean(impacts)),
        "avg_fuel_left": float(np.mean(fuels)),
    }


def rollout_trajectory(env, agent, seed=777):
    state, _ = env.reset(seed=seed)
    traj = {"t": [], "h": [], "v": [], "fuel": [], "action": []}
    step = 0
    while True:
        action = agent.select_action(state, greedy=True)
        traj["t"].append(step * env.DT)
        traj["h"].append(env.h)
        traj["v"].append(env.v)
        traj["fuel"].append(env.fuel)
        traj["action"].append(action)
        state, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            traj["t"].append(step * env.DT)
            traj["h"].append(env.h)
            traj["v"].append(env.v)
            traj["fuel"].append(env.fuel)
            traj["action"].append(0)
            break
    traj["soft"] = terminated and abs(env.v) < env.V_SAFE
    traj["impact"] = abs(env.v)
    return traj


# =========================
# Визуализация
# =========================
def moving_average(data, window=20):
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode="valid")


def save_training_plot(rewards_history, best_episode=None):
    ma = moving_average(rewards_history)
    plt.figure(figsize=(10, 5.6))
    plt.plot(rewards_history, alpha=0.35, label="Награда за эпизод")
    if len(ma):
        plt.plot(range(len(rewards_history) - len(ma), len(rewards_history)), ma,
                 linewidth=2, color="#c0392b", label="Скользящее среднее (20)")
    if best_episode:
        plt.axvline(best_episode - 1, color="#27ae60", linestyle="--", linewidth=1.5,
                    label=f"Лучшая политика (эп. {best_episode})")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.title("Прогресс обучения Double DQN (среда «Мягкая посадка»)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_5_training_history.png"), dpi=160)
    plt.close()


def save_trajectory_plot(traj, v_safe):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(traj["t"], traj["h"], color="#2c3e50", linewidth=2)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Высота")
    axes[0].set_xlabel("Время, с")
    axes[0].set_ylabel("h")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(traj["t"], traj["v"], color="#2980b9", linewidth=2)
    axes[1].axhspan(-v_safe, v_safe, color="#2ecc71", alpha=0.2, label="зона мягкой посадки")
    axes[1].set_title("Вертикальная скорость")
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("v")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(traj["t"], traj["fuel"], color="#e67e22", linewidth=2)
    axes[2].set_title("Остаток топлива")
    axes[2].set_xlabel("Время, с")
    axes[2].set_ylabel("fuel")
    axes[2].grid(True, alpha=0.3)

    verdict = "МЯГКАЯ ПОСАДКА" if traj["soft"] else "ЖЁСТКОЕ КАСАНИЕ"
    fig.suptitle(f"Тестовый полёт обученного агента — {verdict} (|v|={traj['impact']:.3f})")
    plt.tight_layout()
    plt.savefig(TRAJECTORY_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_6_landing.png"), dpi=160)
    plt.close()


def save_landing_frames(traj):
    n = len(traj["h"])
    idx = np.linspace(0, n - 1, 8).astype(int)
    h_max = max(traj["h"]) * 1.1 + 1e-6
    labels = {0: "выкл", 1: "малая", 2: "полная"}

    fig, axes = plt.subplots(1, len(idx), figsize=(14, 4.6))
    for ax, i in zip(axes, idx):
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, h_max)
        ax.set_xticks([])
        h = traj["h"][i]
        a = traj["action"][i]
        # поверхность
        ax.axhspan(0, h_max * 0.02, color="#8d6e63")
        # аппарат
        ax.add_patch(plt.Rectangle((-0.22, h), 0.44, h_max * 0.05,
                                    color="#34495e"))
        # факел тяги
        if a > 0:
            flame_len = h_max * (0.05 + 0.05 * a)
            ax.plot([0, 0], [h, h - flame_len], color="#e74c3c",
                    linewidth=2 + 2 * a)
        ax.set_title(f"t={traj['t'][i]:.1f}с\nh={h:.1f} v={traj['v'][i]:.2f}\nтяга: {labels[a]}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Кадры спуска обученного агента (слева направо — ход посадки)")
    plt.tight_layout()
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_7_landing_frames.png"), dpi=160)
    plt.close()


def save_text_protocol(filename, lines, font_size=11):
    path = os.path.join(PROTOCOL_DIR, filename)
    height = max(3.2, 0.6 + 0.33 * len(lines))
    plt.figure(figsize=(10, height))
    plt.axis("off")
    plt.text(0.02, 0.96, "\n".join(lines), fontsize=font_size, family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_training_log(log_rows):
    with open(TRAINING_LOG_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "epsilon", "soft_landing"])
        writer.writerows(log_rows)


# =========================
# main
# =========================
def main():
    env = SoftLandingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DoubleDQNAgent(state_dim, action_dim)

    print("=== ИНФОРМАЦИЯ О СРЕДЕ ===")
    print("Среда: SoftLanding-v0 (собственная, мягкая посадка аппарата)")
    print(f"Размер состояния: {state_dim} (высота, скорость, топливо)")
    print(f"Количество действий: {action_dim} (0 - выкл, 1 - малая тяга, 2 - полная тяга)")
    print(f"Гравитация: {env.GRAVITY}, тяга: {env.THRUST}, порог мягкой посадки |v|<{env.V_SAFE}")
    print()
    print("=== СТРУКТУРА Q-СЕТИ ===")
    print(agent.policy_net)
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"Всего параметров: {total_params}")
    print(f"Устройство: {device}")
    print()

    print("=== ОБУЧЕНИЕ ===")
    rewards_history, log_rows, best_episode = train(env, agent)
    save_training_log(log_rows)
    torch.save(agent.policy_net.state_dict(), POLICY_PATH)
    print()

    print("=== ОЦЕНКА ОБУЧЕННОГО АГЕНТА ===")
    metrics = evaluate(env, agent, episodes=50)
    print(f"Тестовых эпизодов: {metrics['episodes']}")
    print(f"Доля мягких посадок: {metrics['success_rate']:.1%}")
    print(f"Средняя скорость касания |v|: {metrics['avg_impact']:.3f}")
    print(f"Средний остаток топлива: {metrics['avg_fuel_left']:.2f} / {env.FUEL_START}")
    print()

    traj = rollout_trajectory(env, agent)

    # графики
    save_training_plot(rewards_history, best_episode)
    save_trajectory_plot(traj, env.V_SAFE)
    save_landing_frames(traj)

    # текстовые протоколы
    save_text_protocol(
        "protocol_1_env.png",
        [
            "Среда: SoftLanding-v0 (собственная среда Gymnasium)",
            "Задача: мягкая посадка аппарата при вертикальном спуске",
            "",
            "Состояние (3): высота h, скорость v, топливо f (нормированные)",
            "Действия (3): 0 - двигатель выкл, 1 - малая тяга, 2 - полная тяга",
            f"Гравитация g = {env.GRAVITY}",
            f"Ускорения тяги: {env.THRUST}",
            f"Расход топлива за шаг: {env.FUEL_COST}",
            f"Стартовая высота: {env.H_START}, старт. скорость: {env.V_START}",
            f"Запас топлива: {env.FUEL_START}, лимит шагов: {env.MAX_STEPS}",
            f"Порог мягкой посадки: |v| < {env.V_SAFE}",
            "",
            "Награда:",
            "  - небольшой штраф за каждый шаг (-0.05) и за расход топлива;",
            "  - при касании: 100 - 120*|v| (чем мягче, тем больше очков);",
            "  - бонус +20*доля_остатка_топлива и +50 за |v|<0.6;",
            "  - штраф -50 за таймаут (не сел за лимит шагов).",
        ],
        font_size=10,
    )
    net_lines = str(agent.policy_net).splitlines()
    save_text_protocol(
        "protocol_2_network.png",
        [
            "Алгоритм: Double DQN (PyTorch)",
            "Архитектура Q-сети: 3 -> 128 -> 128 -> 3, активация ReLU",
            "",
            *net_lines,
            "",
            f"Всего обучаемых параметров: {total_params}",
            f"Оптимизатор: Adam, lr={LR}",
            "Функция потерь: SmoothL1 (Huber)",
            f"gamma={GAMMA}, batch={BATCH_SIZE}, replay={MEMORY_SIZE}",
            f"epsilon: {EPS_START} -> {EPS_END} (decay {EPS_DECAY})",
            f"Мягкое обновление целевой сети: tau={TAU} (Polyak)",
            "Лучший чекпойнт выбирается по доле мягких посадок (early stopping)",
            f"Устройство: {device}",
        ],
        font_size=10,
    )
    start_rows = [
        f"эпизод {r[0] + 1:>3d}: reward={r[1]:8.2f}, eps={r[2]:.3f}, "
        f"{'мягкая' if r[3] else 'нет'}"
        for r in log_rows[:12]
    ]
    save_text_protocol(
        "protocol_3_training_start.png",
        ["Начало обучения (первые эпизоды, агент исследует среду):", "", *start_rows],
        font_size=10,
    )
    end_rows = [
        f"эпизод {r[0] + 1:>3d}: reward={r[1]:8.2f}, eps={r[2]:.3f}, "
        f"{'мягкая' if r[3] else 'нет'}"
        for r in log_rows[-12:]
    ]
    save_text_protocol(
        "protocol_4_training_end.png",
        [
            "Конец обучения (последние эпизоды трейна):",
            "В трейне сохраняется разведка ε-greedy (ε=0.05), поэтому отдельные",
            "эпизоды дают 'нет' из-за случайных действий. Качество стратегии",
            "оценивается жадно (без разведки) — см. итоговую оценку ниже.",
            "",
            *end_rows,
            "",
            f"Жадная оценка, доля мягких посадок (50 эп.): {metrics['success_rate']:.1%}",
            f"Средняя скорость касания: {metrics['avg_impact']:.3f}",
            f"Средний остаток топлива: {metrics['avg_fuel_left']:.2f}",
        ],
        font_size=10,
    )

    print("Артефакты сохранены:")
    for path in [
        TRAINING_LOG_PATH, TRAINING_HISTORY_PATH, POLICY_PATH, TRAJECTORY_PATH,
        os.path.join(PROTOCOL_DIR, "protocol_1_env.png"),
        os.path.join(PROTOCOL_DIR, "protocol_2_network.png"),
        os.path.join(PROTOCOL_DIR, "protocol_3_training_start.png"),
        os.path.join(PROTOCOL_DIR, "protocol_4_training_end.png"),
        os.path.join(PROTOCOL_DIR, "protocol_5_training_history.png"),
        os.path.join(PROTOCOL_DIR, "protocol_6_landing.png"),
        os.path.join(PROTOCOL_DIR, "protocol_7_landing_frames.png"),
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

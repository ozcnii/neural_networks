import copy
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPLCONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib_cache")
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)
os.environ.setdefault("XDG_CACHE_HOME", MPLCONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Воспроизводимость
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
DATA_DIR = os.path.join(BASE_DIR, "data")
for d in (ARTIFACTS_DIR, PROTOCOL_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "cpu_load.csv")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")
TRAINING_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.png")
FORECAST_PATH = os.path.join(ARTIFACTS_DIR, "forecast.png")
RESIDUALS_PATH = os.path.join(ARTIFACTS_DIR, "residuals.png")
MULTISTEP_PATH = os.path.join(ARTIFACTS_DIR, "multistep_forecast.png")
BASELINES_PATH = os.path.join(ARTIFACTS_DIR, "baselines.png")
SERIES_PATH = os.path.join(ARTIFACTS_DIR, "series.png")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Параметры задачи
# =========================
N_HOURS = 1800            # длина ряда (75 суток почасовых отсчётов)
DAY = 24
WEEK = 24 * 7
WINDOW = 48               # длина входного окна (двое суток истории)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15           # train / val / test = 0.70 / 0.15 / 0.15
HORIZON = 48              # горизонт рекурсивного multi-step прогноза

# Гиперпараметры модели и обучения
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 120
PATIENCE = 15             # терпение early stopping (эпох без улучшения val-loss)


# =========================
# Генерация своего ряда: загрузка сервера (% CPU)
# =========================
def generate_series():
    """Синтетический, но реалистичный ряд почасовой загрузки сервера (% CPU).

    Состоит из: базового уровня, плавно насыщающегося тренда (рост нагрузки),
    суточной сезонности из двух гармоник (несинусоидальный рабочий день),
    недельной сезонности (спад в выходные) и шума с лёгкой AR(1)-корреляцией.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(N_HOURS, dtype=np.float64)

    base = 42.0

    # Плавно насыщающийся тренд: рост загрузки за период наблюдения (~+12%).
    trend = 12.0 * (1.0 - np.exp(-t / (N_HOURS / 2.0)))

    # Суточная сезонность: две гармоники дают несимметричный «рабочий день».
    phase = 2.0 * np.pi * (t % DAY) / DAY
    daily = 16.0 * np.sin(phase - np.pi / 2.0) + 6.0 * np.sin(2.0 * phase)

    # Недельная сезонность: спад нагрузки в выходные дни.
    day_of_week = (t // DAY) % 7
    weekend = np.where(day_of_week >= 5, -10.0, 0.0)
    weekly = -5.0 * np.sin(2.0 * np.pi * (t % WEEK) / WEEK) + weekend

    # Шум с автокорреляцией (AR(1)): шумовые всплески нагрузки «тянутся».
    noise = np.zeros(N_HOURS)
    eps = rng.normal(0.0, 2.2, size=N_HOURS)
    for i in range(1, N_HOURS):
        noise[i] = 0.6 * noise[i - 1] + eps[i]

    series = base + trend + daily + weekly + noise
    series = np.clip(series, 0.0, 100.0)
    return series.astype(np.float32)


def save_series_csv(series):
    df = pd.DataFrame({"hour": np.arange(len(series)), "cpu_load": series})
    df.to_csv(DATA_PATH, index=False)


# =========================
# Подготовка окон и разбиение по времени
# =========================
def make_windows(series, idx_start, idx_end):
    """Окна, чья ЦЕЛЬ (предсказываемая точка) лежит в [idx_start, idx_end)."""
    xs, ys = [], []
    for target in range(max(idx_start, WINDOW), idx_end):
        xs.append(series[target - WINDOW:target])
        ys.append(series[target])
    x = np.asarray(xs, dtype=np.float32)[:, :, None]   # (N, WINDOW, 1)
    y = np.asarray(ys, dtype=np.float32)[:, None]       # (N, 1)
    return torch.from_numpy(x), torch.from_numpy(y)


def prepare_data(series):
    n = len(series)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    # Нормировка z-score по статистикам ТОЛЬКО train (без утечки из val/test).
    train_raw = series[:train_end]
    mean = float(train_raw.mean())
    std = float(train_raw.std())
    norm = (series - mean) / std

    splits = {
        "train": make_windows(norm, 0, train_end),
        "val": make_windows(norm, train_end, val_end),
        "test": make_windows(norm, val_end, n),
    }
    bounds = {"train_end": train_end, "val_end": val_end, "n": n}
    stats = {"mean": mean, "std": std}
    return norm, splits, bounds, stats


# =========================
# Модель: LSTM + линейная голова
# =========================
class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, WINDOW, hidden)
        last = out[:, -1, :]           # выход последнего шага последовательности
        return self.head(last)         # (B, 1)


# =========================
# Обучение с early stopping по val-loss
# =========================
def train(model, splits):
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(x_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        log_rows.append((epoch, train_loss, val_loss))

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            print(
                f"Эпоха {epoch:>3d}/{MAX_EPOCHS} | "
                f"train MSE={train_loss:.5f} | val MSE={val_loss:.5f}"
                f"{'  <- лучший' if improved else ''}"
            )

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping на эпохе {epoch} "
                  f"(нет улучшения val-loss {PATIENCE} эпох).")
            break

    print(f"\nЛучший чекпойнт: эпоха {best_epoch}, val MSE={best_val:.5f}")
    model.load_state_dict(best_state)
    return log_rows, best_epoch, best_val


# =========================
# Оценка
# =========================
def evaluate_one_step(model, splits, stats):
    """Точечный прогноз на 1 шаг по истинным окнам теста (в % CPU)."""
    x_test, y_test = splits["test"]
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device)).cpu().numpy().ravel()
    true = y_test.numpy().ravel()

    pred = pred * stats["std"] + stats["mean"]
    true = true * stats["std"] + stats["mean"]

    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return true, pred, err, mae, rmse


def evaluate_baselines(series, bounds):
    """Наивные бейзлайны на тех же тестовых точках (в исходных % CPU).

    persistence: прогноз = предыдущее значение (t-1);
    сезонный: прогноз = значение сутки назад (t-24).
    """
    targets = np.arange(bounds["val_end"], bounds["n"])
    true = series[targets]
    results = {}
    for name, lag in (("naive (t-1)", 1), ("сезонный (t-24)", DAY)):
        err = series[targets - lag] - true
        results[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }
    return results


def forecast_multistep(model, norm, bounds, stats, horizon=HORIZON):
    """Замкнутый рекурсивный прогноз на `horizon` шагов от начала теста.

    Берём истинное окно, заканчивающееся в начале теста, и далее кормим
    собственные прогнозы обратно на вход — видно накопление ошибки.
    """
    start = bounds["val_end"]
    window = list(norm[start - WINDOW:start])
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window[-WINDOW:], dtype=torch.float32,
                             device=device).view(1, WINDOW, 1)
            p = float(model(x).item())
            preds.append(p)
            window.append(p)

    preds = np.asarray(preds, dtype=np.float32) * stats["std"] + stats["mean"]
    actual = norm[start:start + horizon] * stats["std"] + stats["mean"]
    return actual, preds


# =========================
# Визуализация
# =========================
def save_series_plot(series, bounds):
    plt.figure(figsize=(12, 5))
    hours = np.arange(len(series))
    plt.plot(hours, series, linewidth=0.8, color="#2c3e50")
    plt.axvspan(0, bounds["train_end"], color="#3498db", alpha=0.08, label="train")
    plt.axvspan(bounds["train_end"], bounds["val_end"], color="#f39c12",
                alpha=0.12, label="val")
    plt.axvspan(bounds["val_end"], bounds["n"], color="#e74c3c",
                alpha=0.10, label="test")
    plt.xlabel("Час наблюдения")
    plt.ylabel("Загрузка CPU, %")
    plt.title("Сгенерированный ряд загрузки сервера (суточная + недельная сезонность)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(SERIES_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_2_series.png"), dpi=160)
    plt.close()


def save_training_plot(log_rows, best_epoch):
    epochs = [r[0] for r in log_rows]
    train_loss = [r[1] for r in log_rows]
    val_loss = [r[2] for r in log_rows]
    plt.figure(figsize=(10, 5.6))
    plt.plot(epochs, train_loss, label="train MSE", color="#2980b9", linewidth=2)
    plt.plot(epochs, val_loss, label="val MSE", color="#c0392b", linewidth=2)
    plt.axvline(best_epoch, color="#27ae60", linestyle="--", linewidth=1.5,
                label=f"лучший чекпойнт (эп. {best_epoch})")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE (нормированные значения)")
    plt.title("Кривая обучения LSTM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_6_training_history.png"), dpi=160)
    plt.close()


def save_forecast_plot(true, pred, mae, rmse):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(true))
    plt.plot(x, true, label="факт", color="#2c3e50", linewidth=1.6)
    plt.plot(x, pred, label="прогноз LSTM (1 шаг)", color="#e67e22",
             linewidth=1.4, linestyle="--")
    plt.xlabel("Час в тестовой выборке")
    plt.ylabel("Загрузка CPU, %")
    plt.title(f"Прогноз vs факт на тесте (one-step)   MAE={mae:.2f}%, RMSE={rmse:.2f}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FORECAST_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_7_forecast.png"), dpi=160)
    plt.close()


def save_residuals_plot(err):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(err, color="#8e44ad", linewidth=0.9)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Остатки прогноза по времени")
    axes[0].set_xlabel("Час в тестовой выборке")
    axes[0].set_ylabel("Ошибка (прогноз - факт), %")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(err, bins=30, color="#9b59b6", alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(float(np.mean(err)), color="#c0392b", linewidth=1.5,
                    label=f"среднее={np.mean(err):.2f}%")
    axes[1].set_title("Распределение остатков")
    axes[1].set_xlabel("Ошибка, %")
    axes[1].set_ylabel("Частота")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Анализ остатков one-step прогноза на тесте")
    plt.tight_layout()
    plt.savefig(RESIDUALS_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_8_residuals.png"), dpi=160)
    plt.close()


def save_multistep_plot(actual, preds):
    plt.figure(figsize=(11, 5))
    x = np.arange(len(actual))
    plt.plot(x, actual, label="факт", color="#2c3e50", linewidth=1.8, marker="o",
             markersize=3)
    plt.plot(x, preds, label="рекурсивный прогноз LSTM", color="#c0392b",
             linewidth=1.8, linestyle="--", marker="s", markersize=3)
    step_mae = float(np.mean(np.abs(preds - actual)))
    plt.xlabel("Шаг прогноза вперёд, ч")
    plt.ylabel("Загрузка CPU, %")
    plt.title(f"Рекурсивный прогноз на {len(actual)} ч вперёд (замкнутый цикл)   "
              f"MAE={step_mae:.2f}%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MULTISTEP_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_9_multistep.png"), dpi=160)
    plt.close()
    return step_mae


def save_baselines_plot(scores):
    names = list(scores.keys())
    maes = [scores[n]["mae"] for n in names]
    rmses = [scores[n]["rmse"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, maes, w, label="MAE", color="#2980b9")
    plt.bar(x + w / 2, rmses, w, label="RMSE", color="#e67e22")
    top = max(maes + rmses)
    for xi, v in zip(x - w / 2, maes):
        plt.text(xi, v + top * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, rmses):
        plt.text(xi, v + top * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, names)
    plt.ylabel("Ошибка, % CPU")
    plt.ylim(0, top * 1.15)
    plt.title("Сравнение LSTM с наивными бейзлайнами (one-step, тест)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASELINES_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_10_baselines.png"), dpi=160)
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
    df = pd.DataFrame(log_rows, columns=["epoch", "train_mse", "val_mse"])
    df.to_csv(TRAINING_LOG_PATH, index=False)


# =========================
# main
# =========================
def main():
    series = generate_series()
    save_series_csv(series)
    norm, splits, bounds, stats = prepare_data(series)

    n_train = splits["train"][0].shape[0]
    n_val = splits["val"][0].shape[0]
    n_test = splits["test"][0].shape[0]

    print("=== ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")
    print("Задача: прогноз почасовой загрузки сервера (% CPU) по окну истории")
    print(f"Длина ряда: {len(series)} ч (~{len(series)//DAY} сут)")
    print(f"Диапазон значений: {series.min():.1f}..{series.max():.1f} % CPU "
          f"(среднее {series.mean():.1f})")
    print(f"Длина окна: {WINDOW} ч, цель: следующее значение")
    print(f"Разбиение по времени: train={n_train}, val={n_val}, test={n_test} окон")
    print(f"Нормировка z-score по train: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    print()

    model = LSTMForecaster().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("=== СТРУКТУРА МОДЕЛИ ===")
    print(model)
    print(f"Всего обучаемых параметров: {total_params}")
    print(f"Устройство: {device}")
    print()

    print("=== ОБУЧЕНИЕ ===")
    log_rows, best_epoch, best_val = train(model, splits)
    save_training_log(log_rows)
    torch.save(model.state_dict(), MODEL_PATH)
    print()

    print("=== ОЦЕНКА НА ТЕСТЕ ===")
    true, pred, err, mae, rmse = evaluate_one_step(model, splits, stats)
    print(f"One-step прогноз ({n_test} точек):")
    print(f"  MAE  = {mae:.3f} % CPU")
    print(f"  RMSE = {rmse:.3f} % CPU")
    print(f"  средний остаток = {np.mean(err):.3f} % CPU (смещение)")

    actual_ms, preds_ms = forecast_multistep(model, norm, bounds, stats)
    ms_mae = float(np.mean(np.abs(preds_ms - actual_ms)))
    print(f"Рекурсивный прогноз на {HORIZON} ч вперёд: MAE = {ms_mae:.3f} % CPU")
    print()

    baselines = evaluate_baselines(series, bounds)
    scores = {"LSTM": {"mae": mae, "rmse": rmse}, **baselines}
    print("Сравнение с наивными бейзлайнами (one-step, % CPU):")
    print(f"  {'модель':<16}{'MAE':>9}{'RMSE':>9}")
    for name, sc in scores.items():
        print(f"  {name:<16}{sc['mae']:>9.3f}{sc['rmse']:>9.3f}")
    best_base = min(b["mae"] for b in baselines.values())
    print(f"LSTM снижает MAE в {best_base / mae:.1f} раза относительно "
          f"лучшего бейзлайна.")
    print()

    # графики
    save_series_plot(series, bounds)
    save_training_plot(log_rows, best_epoch)
    save_forecast_plot(true, pred, mae, rmse)
    save_residuals_plot(err)
    save_multistep_plot(actual_ms, preds_ms)
    save_baselines_plot(scores)

    # текстовые протоколы
    save_text_protocol(
        "protocol_1_dataset.png",
        [
            "Задача: прогнозирование временного ряда с помощью LSTM",
            "Ряд: почасовая загрузка сервера (% CPU), синтетический",
            "",
            f"Длина ряда: {len(series)} ч (~{len(series)//DAY} суток)",
            "Состав ряда:",
            "  - базовый уровень ~42% + насыщающийся тренд (рост нагрузки);",
            "  - суточная сезонность (2 гармоники, период 24 ч);",
            "  - недельная сезонность (спад в выходные, период 168 ч);",
            "  - шум с автокорреляцией AR(1).",
            "",
            f"Длина входного окна L: {WINDOW} ч",
            "Цель: следующее значение ряда (one-step)",
            "Разбиение по времени (без перемешивания):",
            f"  train {int(TRAIN_FRAC*100)}% / val {int(VAL_FRAC*100)}% / "
            f"test {int((1-TRAIN_FRAC-VAL_FRAC)*100)}%",
            f"  окон: train={n_train}, val={n_val}, test={n_test}",
            f"Нормировка z-score по train: mean={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}",
        ],
        font_size=10,
    )
    net_lines = str(model).splitlines()
    save_text_protocol(
        "protocol_3_network.png",
        [
            "Модель: LSTM-прогнозатор (PyTorch)",
            f"Архитектура: nn.LSTM(input=1, hidden={HIDDEN_SIZE}, "
            f"layers={NUM_LAYERS}) -> Linear({HIDDEN_SIZE}, 1)",
            "Вход: окно (L, 1); берётся выход последнего шага последовательности.",
            "",
            *net_lines,
            "",
            f"Всего обучаемых параметров: {total_params}",
            "Функция потерь: MSE",
            f"Оптимизатор: Adam, lr={LR}",
            f"Batch size: {BATCH_SIZE}, макс. эпох: {MAX_EPOCHS}",
            f"Early stopping по val-loss, patience={PATIENCE}",
            f"Устройство: {device}",
        ],
        font_size=10,
    )
    start_rows = [
        f"эпоха {r[0]:>3d}: train MSE={r[1]:.5f}, val MSE={r[2]:.5f}"
        for r in log_rows[:12]
    ]
    save_text_protocol(
        "protocol_4_training_start.png",
        ["Начало обучения (первые эпохи):", "", *start_rows],
        font_size=10,
    )
    end_rows = [
        f"эпоха {r[0]:>3d}: train MSE={r[1]:.5f}, val MSE={r[2]:.5f}"
        for r in log_rows[-12:]
    ]
    save_text_protocol(
        "protocol_5_training_end.png",
        [
            "Конец обучения (последние эпохи):",
            "",
            *end_rows,
            "",
            f"Лучший чекпойнт: эпоха {best_epoch}, val MSE={best_val:.5f}",
            "Итоговые метрики на тесте (в исходных % CPU):",
            f"  one-step:  MAE={mae:.3f}, RMSE={rmse:.3f}",
            f"  multi-step ({HORIZON} ч):  MAE={ms_mae:.3f}",
            "",
            "Сравнение с наивными бейзлайнами (one-step, % CPU):",
            f"  {'модель':<16}{'MAE':>8}{'RMSE':>8}",
            *[f"  {name:<16}{sc['mae']:>8.3f}{sc['rmse']:>8.3f}"
              for name, sc in scores.items()],
        ],
        font_size=10,
    )

    print("Артефакты сохранены:")
    for path in [
        DATA_PATH, TRAINING_LOG_PATH, MODEL_PATH,
        SERIES_PATH, TRAINING_HISTORY_PATH, FORECAST_PATH,
        RESIDUALS_PATH, MULTISTEP_PATH, BASELINES_PATH,
        os.path.join(PROTOCOL_DIR, "protocol_1_dataset.png"),
        os.path.join(PROTOCOL_DIR, "protocol_2_series.png"),
        os.path.join(PROTOCOL_DIR, "protocol_3_network.png"),
        os.path.join(PROTOCOL_DIR, "protocol_4_training_start.png"),
        os.path.join(PROTOCOL_DIR, "protocol_5_training_end.png"),
        os.path.join(PROTOCOL_DIR, "protocol_6_training_history.png"),
        os.path.join(PROTOCOL_DIR, "protocol_7_forecast.png"),
        os.path.join(PROTOCOL_DIR, "protocol_8_residuals.png"),
        os.path.join(PROTOCOL_DIR, "protocol_9_multistep.png"),
        os.path.join(PROTOCOL_DIR, "protocol_10_baselines.png"),
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

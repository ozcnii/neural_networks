"""Лабораторная работа 8. AutoML: подбор моделей и гиперпараметров.

Сравнение самописной нейронной сети из ЛР-2 (бинарная классификация риска
отчисления студента) с автоматически подобранными решениями H2O AutoML и FEDOT
на едином стратифицированном сплите и одинаковом наборе метрик.
"""

import csv
import io
import json
import logging
import os
import random
import warnings
from contextlib import redirect_stderr, redirect_stdout

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
DATASET_PATH = os.path.join(DATA_DIR, "student_dropout_risk_dataset.csv")

FEATURE_COLUMNS = [
    "average_grade",
    "attendance_percent",
    "academic_debts",
    "study_hours_per_week",
    "class_activity_score",
]
TARGET_COLUMN = "high_dropout_risk"

# Бюджеты AutoML заданы "в штуках", а не во времени -> воспроизводимость.
H2O_MAX_MODELS = 20
H2O_NFOLDS = 5
FEDOT_GENERATIONS = 5
FEDOT_TIMEOUT_MIN = 3
CV_FOLDS = 5

# Гиперпараметры самописной НС взяты без изменений из ЛР-2.
NN_HIDDEN = 6
NN_LEARNING_RATE = 0.45
NN_MAX_EPOCHS = 5000
NN_ERROR_THRESHOLD = 0.012

# Бюджет Optuna для подбора гиперпараметров самой НС (буквальная трактовка задания).
# В поиске используется укороченное число эпох ради скорости; лучший конфиг затем
# переобучается на полном NN_MAX_EPOCHS. Подбор детерминирован (TPESampler с seed).
HPO_TRIALS = 20
HPO_CV_FOLDS = 3
HPO_TUNE_EPOCHS = 1000

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Данные и единый сплит
# --------------------------------------------------------------------------- #
def load_dataset():
    frame = pd.read_csv(DATASET_PATH)
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=int)
    return frame, features, targets


def normalize_features(features):
    """Доменная нормализация из ЛР-2 (константы, без утечки из данных)."""
    normalized = np.zeros_like(features, dtype=float)
    normalized[:, 0] = (features[:, 0] - 2.0) / 3.0
    normalized[:, 1] = features[:, 1] / 100.0
    normalized[:, 2] = features[:, 2] / 7.0
    normalized[:, 3] = features[:, 3] / 28.0
    normalized[:, 4] = features[:, 4] / 10.0
    return normalized


# --------------------------------------------------------------------------- #
# Самописная нейронная сеть из ЛР-2 (перенесена без изменений логики)
# --------------------------------------------------------------------------- #
def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(activated_value):
    return activated_value * (1.0 - activated_value)


class Neuron:
    def __init__(self, input_count, rng):
        self.weights = rng.uniform(-0.6, 0.6, size=input_count)
        self.bias = 0.0
        self.output = 0.0

    def forward(self, inputs):
        net = np.dot(self.weights, inputs) + self.bias
        self.output = sigmoid(net)
        return self.output

    def update_weights(self, delta, inputs, learning_rate):
        self.weights += learning_rate * delta * inputs
        self.bias += learning_rate * delta


class Layer:
    def __init__(self, input_count, neuron_count, rng):
        self.neurons = [Neuron(input_count, rng) for _ in range(neuron_count)]
        self.outputs = np.array([])

    def forward(self, inputs):
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def update_weights(self, deltas, inputs, learning_rate):
        for neuron, delta in zip(self.neurons, deltas):
            neuron.update_weights(delta, inputs, learning_rate)

    def weights_matrix(self):
        return np.array([neuron.weights for neuron in self.neurons])


class NeuralNetwork:
    def __init__(self, input_count, layer_sizes, learning_rate=NN_LEARNING_RATE, seed=SEED):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.layers = []
        previous_size = input_count
        for size in layer_sizes:
            self.layers.append(Layer(previous_size, size, rng))
            previous_size = size

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, inputs, targets):
        deltas = [None] * len(self.layers)
        last_layer = self.layers[-1]
        deltas[-1] = (targets - last_layer.outputs) * sigmoid_derivative(last_layer.outputs)
        for layer_index in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            error = deltas[layer_index + 1] @ next_layer.weights_matrix()
            deltas[layer_index] = error * sigmoid_derivative(current_layer.outputs)
        for layer_index, layer in enumerate(self.layers):
            layer_inputs = inputs if layer_index == 0 else self.layers[layer_index - 1].outputs
            layer.update_weights(deltas[layer_index], layer_inputs, self.learning_rate)

    def predict_probability(self, inputs):
        return float(self.forward(inputs)[0])

    def fit(self, features, targets, max_epochs=NN_MAX_EPOCHS, error_threshold=NN_ERROR_THRESHOLD):
        for epoch in range(1, max_epochs + 1):
            mse = 0.0
            for inputs, target in zip(features, targets):
                output = self.forward(inputs)
                self.backward(inputs, np.array([target], dtype=float))
                mse += float((target - output[0]) ** 2)
            mse /= len(features)
            if mse < error_threshold:
                break
        return mse, epoch

    def predict_proba(self, features):
        return np.array([self.predict_probability(row) for row in features])


def train_nn(x_train, y_train, hidden=NN_HIDDEN, learning_rate=NN_LEARNING_RATE,
             max_epochs=NN_MAX_EPOCHS, seed=SEED):
    network = NeuralNetwork(
        input_count=len(FEATURE_COLUMNS),
        layer_sizes=[hidden, 1],
        learning_rate=learning_rate,
        seed=seed,
    )
    final_mse, final_epoch = network.fit(x_train, y_train, max_epochs=max_epochs)
    return network, final_mse, final_epoch


def tune_nn_optuna(x_train, y_train):
    """Подбор гиперпараметров самой НС (hidden, learning_rate) через Optuna (TPE).

    Целевая метрика - 3-fold CV accuracy ТОЛЬКО на train (без утечки из теста).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [4, 6, 8, 10, 12])
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.9, log=True)
        scores = []
        for tr_idx, val_idx in skf.split(x_train, y_train):
            network, _, _ = train_nn(
                x_train[tr_idx], y_train[tr_idx],
                hidden=hidden, learning_rate=learning_rate, max_epochs=HPO_TUNE_EPOCHS,
            )
            y_prob = network.predict_proba(x_train[val_idx])
            scores.append(accuracy_score(y_train[val_idx], (y_prob >= 0.5).astype(int)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# --------------------------------------------------------------------------- #
# Протокольные PNG (текст) — по образцу save_text_protocol из ЛР-2/ЛР-6
# --------------------------------------------------------------------------- #
def save_text_protocol(filename, lines, font_size=11):
    path = os.path.join(PROTOCOL_DIR, filename)
    height = max(3.2, 0.6 + 0.33 * len(lines))
    plt.figure(figsize=(11, height))
    plt.axis("off")
    plt.text(0.01, 0.98, "\n".join(lines), fontsize=font_size, family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_fig(fig, *names):
    for name in names:
        fig.savefig(name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# H2O AutoML
# --------------------------------------------------------------------------- #
def run_h2o(train_df, test_df, y_test):
    import h2o
    from h2o.automl import H2OAutoML

    # nthreads=1 обязателен для воспроизводимости: модели Deep Learning в H2O
    # при многопоточности используют Hogwild (асинхронный SGD) и недетерминированы.
    h2o.init(nthreads=1, max_mem_size="3G")
    h2o.no_progress()

    train_hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)
    train_hf[TARGET_COLUMN] = train_hf[TARGET_COLUMN].asfactor()
    test_hf[TARGET_COLUMN] = test_hf[TARGET_COLUMN].asfactor()

    aml = H2OAutoML(max_models=H2O_MAX_MODELS, seed=SEED, nfolds=H2O_NFOLDS, sort_metric="AUC")
    aml.train(x=FEATURE_COLUMNS, y=TARGET_COLUMN, training_frame=train_hf)

    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard.to_csv(os.path.join(ARTIFACTS_DIR, "h2o_leaderboard.csv"), index=False)

    leader = aml.leader
    preds = leader.predict(test_hf).as_data_frame()
    y_prob = preds["p1"].to_numpy()
    metrics = compute_metrics(y_test, y_prob)

    leader_id = leader.model_id
    leader_row = leaderboard.iloc[0].to_dict()
    cv_auc = float(leader_row.get("auc", float("nan")))

    # Variable importance: лидер часто StackedEnsemble без varimp -> берём
    # лучшую базовую модель из лидерборда, у которой varimp доступен.
    varimp = None
    varimp_model_id = None
    for model_id in leaderboard["model_id"].tolist():
        model = h2o.get_model(model_id)
        try:
            table = model.varimp(use_pandas=True)
        except Exception:
            table = None
        if table is not None and len(table) > 0:
            varimp = table
            varimp_model_id = model_id
            break

    model_path = h2o.save_model(leader, path=ARTIFACTS_DIR, force=True)

    result = {
        "metrics": metrics,
        "leader_id": leader_id,
        "cv_auc": cv_auc,
        "leaderboard": leaderboard,
        "n_models": len(leaderboard),
        "y_prob": y_prob,
        "varimp": varimp,
        "varimp_model_id": varimp_model_id,
        "model_path": model_path,
    }

    h2o.cluster().shutdown()
    return result


# --------------------------------------------------------------------------- #
# FEDOT
# --------------------------------------------------------------------------- #
def run_fedot(x_train_raw, y_train, x_test_raw, y_test):
    logging.disable(logging.WARNING)
    from fedot.api.main import Fedot

    model = Fedot(
        problem="classification",
        timeout=FEDOT_TIMEOUT_MIN,
        num_of_generations=FEDOT_GENERATIONS,
        seed=SEED,
        logging_level=logging.CRITICAL,
        show_progress=False,
        n_jobs=1,
    )

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model.fit(features=x_train_raw, target=y_train)
        y_prob = np.asarray(model.predict_proba(features=x_test_raw)).reshape(-1)

    logging.disable(logging.NOTSET)
    metrics = compute_metrics(y_test, y_prob)

    pipeline = model.current_pipeline
    nodes = [node.name for node in pipeline.nodes]
    structure = pipeline.descriptive_id

    pipeline_path = os.path.join(ARTIFACTS_DIR, "fedot_pipeline.json")
    try:
        pipeline.save(path=pipeline_path)
    except Exception:
        pipeline_path = None

    return {
        "metrics": metrics,
        "y_prob": y_prob,
        "nodes": nodes,
        "structure": structure,
        "pipeline_path": pipeline_path,
    }


# --------------------------------------------------------------------------- #
# Кросс-валидация самописной НС (stratified k-fold)
# --------------------------------------------------------------------------- #
def cross_validate_nn(features_norm, targets):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = {"accuracy": [], "roc_auc": [], "f1": []}
    for train_idx, val_idx in skf.split(features_norm, targets):
        network, _, _ = train_nn(features_norm[train_idx], targets[train_idx])
        y_prob = network.predict_proba(features_norm[val_idx])
        y_true = targets[val_idx]
        y_pred = (y_prob >= 0.5).astype(int)
        fold_metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_true, y_prob))
        fold_metrics["f1"].append(f1_score(y_true, y_pred))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in fold_metrics.items()}


# --------------------------------------------------------------------------- #
# Графики
# --------------------------------------------------------------------------- #
MODEL_COLORS = ["#4c72b0", "#c44e52", "#dd8452", "#55a868"]


def plot_metric_bars(results):
    metric_keys = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
    models = list(results.keys())
    n = len(models)

    x = np.arange(len(metric_keys))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, model in enumerate(models):
        values = [results[model]["metrics"][k] for k in metric_keys]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model, color=MODEL_COLORS[i % len(MODEL_COLORS)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение метрик на тестовой выборке: самописная НС vs AutoML")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, os.path.join(ARTIFACTS_DIR, "metrics_comparison.png"),
             os.path.join(PROTOCOL_DIR, "protocol_7_metrics_bar.png"))


def plot_roc(results, y_test):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for i, (model, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
        auc = data["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[i % len(MODEL_COLORS)],
                label=f"{model} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые на тестовой выборке")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(ARTIFACTS_DIR, "roc_curves.png"),
             os.path.join(PROTOCOL_DIR, "protocol_8_roc.png"))


def plot_confusion(results):
    models = list(results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2))
    for ax, model in zip(axes, models):
        cm = np.array(results[model]["metrics"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(model, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Предсказано"); ax.set_ylabel("Факт")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Матрицы ошибок (0 - низкий риск, 1 - высокий риск)", fontsize=12)
    save_fig(fig, os.path.join(ARTIFACTS_DIR, "confusion_matrices.png"),
             os.path.join(PROTOCOL_DIR, "protocol_9_confusion.png"))


def plot_feature_importance(varimp, varimp_model_id):
    fig, ax = plt.subplots(figsize=(9, 5))
    if varimp is not None:
        names = varimp["variable"].tolist()[::-1]
        scaled = varimp["scaled_importance"].tolist()[::-1]
        ax.barh(names, scaled, color="#4c72b0")
        ax.set_xlabel("Относительная важность (scaled)")
        ax.set_title(f"Важность признаков (H2O, модель {varimp_model_id})")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Variable importance недоступен\n(лидер — ансамбль без varimp)",
                ha="center", va="center")
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, os.path.join(ARTIFACTS_DIR, "feature_importance.png"),
             os.path.join(PROTOCOL_DIR, "protocol_10_feature_importance.png"))


def plot_hpo_history(study):
    values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(values)
    trials = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trials, values, "o", color="#4c72b0", alpha=0.5, label="CV accuracy пробы")
    ax.plot(trials, best_so_far, "-", color="#c44e52", linewidth=2, label="Лучшее значение")
    ax.set_xlabel("Номер пробы Optuna")
    ax.set_ylabel(f"{HPO_CV_FOLDS}-fold CV accuracy (train)")
    ax.set_title("Подбор гиперпараметров самописной НС (Optuna, TPE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(ARTIFACTS_DIR, "optuna_history.png"),
             os.path.join(PROTOCOL_DIR, "protocol_12_optuna_history.png"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    for directory in (DATA_DIR, ARTIFACTS_DIR, PROTOCOL_DIR):
        os.makedirs(directory, exist_ok=True)

    print("Лабораторная работа 8. AutoML: подбор моделей и гиперпараметров")
    print("Сравнение самописной НС (ЛР-2) с H2O AutoML и FEDOT\n")

    frame, features, targets = load_dataset()
    features_norm = normalize_features(features)
    n_pos = int(targets.sum())
    n_neg = int(len(targets) - n_pos)
    print(f"1. Датасет: {len(frame)} строк, признаков: {len(FEATURE_COLUMNS)}")
    print(f"   Классы: низкий риск={n_neg}, высокий риск={n_pos}\n")

    # Единый стратифицированный сплит для ВСЕХ моделей.
    indices = np.arange(len(targets))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=SEED
    )
    x_train_raw, x_test_raw = features[train_idx], features[test_idx]
    x_train_norm, x_test_norm = features_norm[train_idx], features_norm[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]
    print(f"2. Единый сплит (стратиф., seed={SEED}): train={len(train_idx)}, test={len(test_idx)}\n")

    train_df = pd.DataFrame(x_train_raw, columns=FEATURE_COLUMNS)
    train_df[TARGET_COLUMN] = y_train
    test_df = pd.DataFrame(x_test_raw, columns=FEATURE_COLUMNS)
    test_df[TARGET_COLUMN] = y_test

    results = {}

    # --- Бейзлайн: самописная НС из ЛР-2 ---
    print("3. Обучение самописной НС (5 -> 6 -> 1, sigmoid, MSE, backprop)...")
    nn_network, nn_mse, nn_epoch = train_nn(x_train_norm, y_train)
    nn_prob = nn_network.predict_proba(x_test_norm)
    nn_metrics = compute_metrics(y_test, nn_prob)
    results["Самописная НС"] = {"metrics": nn_metrics, "y_prob": nn_prob}
    print(f"   Остановка на эпохе {nn_epoch}, MSE={nn_mse:.6f}")
    print(f"   test accuracy={nn_metrics['accuracy']:.4f}, AUC={nn_metrics['roc_auc']:.4f}\n")

    # --- HPO самописной НС через Optuna (буквальный "подбор гиперпараметров") ---
    print(f"3b. Подбор гиперпараметров НС через Optuna ({HPO_TRIALS} проб, TPE)...")
    study = tune_nn_optuna(x_train_norm, y_train)
    best_params = study.best_params
    tuned_network, tuned_mse, tuned_epoch = train_nn(
        x_train_norm, y_train,
        hidden=best_params["hidden"], learning_rate=best_params["learning_rate"],
    )
    tuned_prob = tuned_network.predict_proba(x_test_norm)
    tuned_metrics = compute_metrics(y_test, tuned_prob)
    results["НС+Optuna"] = {"metrics": tuned_metrics, "y_prob": tuned_prob}
    print(f"   Лучшие гиперпараметры: hidden={best_params['hidden']}, "
          f"lr={best_params['learning_rate']:.4f} (CV accuracy={study.best_value:.4f})")
    print(f"   test accuracy={tuned_metrics['accuracy']:.4f}, "
          f"AUC={tuned_metrics['roc_auc']:.4f}\n")

    # --- H2O AutoML ---
    print(f"4. H2O AutoML (max_models={H2O_MAX_MODELS}, nfolds={H2O_NFOLDS}, seed={SEED})...")
    h2o_result = run_h2o(train_df, test_df, y_test)
    results["H2O AutoML"] = {"metrics": h2o_result["metrics"], "y_prob": h2o_result["y_prob"]}
    print(f"   Лидер: {h2o_result['leader_id']} (перебрано моделей: {h2o_result['n_models']})")
    print(f"   test accuracy={h2o_result['metrics']['accuracy']:.4f}, "
          f"AUC={h2o_result['metrics']['roc_auc']:.4f}, CV-AUC={h2o_result['cv_auc']:.4f}\n")

    # --- FEDOT ---
    print(f"5. FEDOT (problem=classification, generations={FEDOT_GENERATIONS}, seed={SEED})...")
    fedot_result = run_fedot(x_train_raw, y_train, x_test_raw, y_test)
    results["FEDOT"] = {"metrics": fedot_result["metrics"], "y_prob": fedot_result["y_prob"]}
    print(f"   Пайплайн: {' -> '.join(fedot_result['nodes'])}")
    print(f"   test accuracy={fedot_result['metrics']['accuracy']:.4f}, "
          f"AUC={fedot_result['metrics']['roc_auc']:.4f}\n")

    # --- Кросс-валидация НС ---
    print(f"6. Stratified {CV_FOLDS}-fold CV для самописной НС...")
    nn_cv = cross_validate_nn(features_norm, targets)
    print(f"   CV accuracy={nn_cv['accuracy'][0]:.4f}±{nn_cv['accuracy'][1]:.4f}, "
          f"CV AUC={nn_cv['roc_auc'][0]:.4f}±{nn_cv['roc_auc'][1]:.4f}\n")

    # --- Сохранение метрик ---
    metrics_export = {
        "split": {"train": int(len(train_idx)), "test": int(len(test_idx)), "seed": SEED},
        "models": {name: data["metrics"] for name, data in results.items()},
        "h2o": {"leader_id": h2o_result["leader_id"], "cv_auc": h2o_result["cv_auc"],
                "n_models": h2o_result["n_models"], "varimp_model": h2o_result["varimp_model_id"]},
        "fedot": {"nodes": fedot_result["nodes"], "structure": fedot_result["structure"]},
        "nn_cv": nn_cv,
        "nn_final": {"mse": nn_mse, "epoch": int(nn_epoch)},
        "nn_hpo": {
            "trials": HPO_TRIALS,
            "best_params": best_params,
            "best_cv_accuracy": float(study.best_value),
            "tuned_epoch": int(tuned_epoch),
            "tuned_mse": float(tuned_mse),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_export, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "comparison.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "accuracy", "roc_auc", "f1", "precision", "recall"])
        for name, data in results.items():
            m = data["metrics"]
            writer.writerow([name, f"{m['accuracy']:.4f}", f"{m['roc_auc']:.4f}",
                             f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"])

    # --- Графики ---
    print("7. Построение графиков и протокольных PNG...")
    plot_metric_bars(results)
    plot_roc(results, y_test)
    plot_confusion(results)
    plot_feature_importance(h2o_result["varimp"], h2o_result["varimp_model_id"])
    plot_hpo_history(study)

    # --- Протокольные текстовые PNG ---
    save_text_protocol("protocol_1_task_dataset.png", [
        "ЛР-8. AutoML: автоматический подбор моделей и гиперпараметров.",
        "",
        f"Базовая задача (из ЛР-2): бинарная классификация риска отчисления студента.",
        f"Датасет: {os.path.basename(DATASET_PATH)} ({len(frame)} строк).",
        f"Признаки ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}.",
        f"Цель: {TARGET_COLUMN} (0 - низкий риск, 1 - высокий риск).",
        f"Баланс классов: низкий={n_neg}, высокий={n_pos}.",
        "",
        f"Единый стратифицированный сплит (seed={SEED}): "
        f"train={len(train_idx)}, test={len(test_idx)}.",
        "Один и тот же сплит используется для НС, H2O AutoML и FEDOT.",
    ], font_size=11)

    save_text_protocol("protocol_2_baseline_nn.png", [
        "Базовая модель - самописная нейросеть из ЛР-2 (NumPy, с нуля).",
        "",
        "Архитектура: 5 -> 6 -> 1, активация sigmoid, ошибка MSE.",
        "Обучение: обратное распространение ошибки, lr=0.45.",
        f"Нормализация признаков: доменные константы (как в ЛР-2).",
        f"Остановка: эпоха {nn_epoch}, MSE={nn_mse:.6f}.",
        "",
        "Метрики на тестовой выборке:",
        f"  accuracy  = {nn_metrics['accuracy']:.4f}",
        f"  ROC-AUC   = {nn_metrics['roc_auc']:.4f}",
        f"  F1        = {nn_metrics['f1']:.4f}",
        f"  precision = {nn_metrics['precision']:.4f}",
        f"  recall    = {nn_metrics['recall']:.4f}",
        f"  CV ({CV_FOLDS}-fold): accuracy={nn_cv['accuracy'][0]:.4f}+/-{nn_cv['accuracy'][1]:.4f}, "
        f"AUC={nn_cv['roc_auc'][0]:.4f}+/-{nn_cv['roc_auc'][1]:.4f}",
    ], font_size=11)

    save_text_protocol("protocol_3_automl_libs.png", [
        "AutoML - автоматизация выбора признаков, моделей, гиперпараметров и",
        "ансамблей. Экономит ручной перебор и часто достигает качества",
        "сопоставимого с тщательно настроенной моделью.",
        "",
        "H2O AutoML (основная): обучает GLM/GBM/DRF/XGBoost/Deep Learning,",
        "  перебирает их гиперпараметры, строит стек-ансамбли и ранжирует",
        f"  модели в лидерборде. Бюджет: max_models={H2O_MAX_MODELS}, nfolds={H2O_NFOLDS}.",
        "",
        "FEDOT (ИТМО): эволюционный поиск структуры пайплайна (граф из",
        f"  предобработки и моделей). Бюджет: {FEDOT_GENERATIONS} поколений.",
        "",
        "AutoKeras не используется: основан на TensorFlow, чья установка на",
        "  arm64 / Python 3.9 нестабильна; исключён осознанно, а не пропущен.",
    ], font_size=11)

    leaderboard = h2o_result["leaderboard"]
    lb_cols = [c for c in ["model_id", "auc", "logloss", "aucpr", "mean_per_class_error"]
               if c in leaderboard.columns]
    lb_view = leaderboard[lb_cols].head(12).copy()
    for col in lb_cols:
        if col != "model_id":
            lb_view[col] = lb_view[col].map(lambda v: f"{v:.4f}")
    lb_lines = lb_view.to_string(index=False).split("\n")
    save_text_protocol("protocol_4_h2o_leaderboard.png", [
        f"Лидерборд H2O AutoML (перебрано моделей: {h2o_result['n_models']}).",
        "Метрики ниже - кросс-валидационные (nfolds=5), сортировка по AUC.",
        "",
        *lb_lines,
        "",
        f"Лучшая модель (лидер): {h2o_result['leader_id']}",
        f"CV-AUC лидера: {h2o_result['cv_auc']:.4f}",
        f"Метрики лидера на тесте: accuracy={h2o_result['metrics']['accuracy']:.4f}, "
        f"AUC={h2o_result['metrics']['roc_auc']:.4f}, F1={h2o_result['metrics']['f1']:.4f}",
    ], font_size=9)

    save_text_protocol("protocol_5_fedot_pipeline.png", [
        "FEDOT: найденный эволюционным поиском пайплайн.",
        "",
        f"Узлы пайплайна: {' -> '.join(fedot_result['nodes'])}",
        "",
        "Структура (descriptive_id):",
        *[fedot_result["structure"][i:i + 90] for i in range(0, min(len(fedot_result["structure"]), 540), 90)],
        "",
        "Метрики на тестовой выборке:",
        f"  accuracy  = {fedot_result['metrics']['accuracy']:.4f}",
        f"  ROC-AUC   = {fedot_result['metrics']['roc_auc']:.4f}",
        f"  F1        = {fedot_result['metrics']['f1']:.4f}",
        f"  precision = {fedot_result['metrics']['precision']:.4f}",
        f"  recall    = {fedot_result['metrics']['recall']:.4f}",
    ], font_size=10)

    header = f"{'Модель':<16}{'Accuracy':>10}{'ROC-AUC':>10}{'F1':>10}{'Precision':>12}{'Recall':>10}"
    table_lines = [header, "-" * len(header)]
    for name, data in results.items():
        m = data["metrics"]
        table_lines.append(
            f"{name:<16}{m['accuracy']:>10.4f}{m['roc_auc']:>10.4f}{m['f1']:>10.4f}"
            f"{m['precision']:>12.4f}{m['recall']:>10.4f}"
        )
    save_text_protocol("protocol_6_comparison.png", [
        "Сравнение моделей на едином тестовом сплите (одинаковые метрики):",
        "",
        *table_lines,
        "",
        f"Тестовая выборка мала (~{len(test_idx)} объектов) -> метрики зашумлены.",
        f"Для устойчивости приведена {CV_FOLDS}-fold CV самописной НС:",
        f"  accuracy={nn_cv['accuracy'][0]:.4f}+/-{nn_cv['accuracy'][1]:.4f}, "
        f"AUC={nn_cv['roc_auc'][0]:.4f}+/-{nn_cv['roc_auc'][1]:.4f}, "
        f"F1={nn_cv['f1'][0]:.4f}+/-{nn_cv['f1'][1]:.4f}",
        "H2O оценивает модели встроенной CV (nfolds=5), FEDOT - внутренней CV композера.",
    ], font_size=10)

    save_text_protocol("protocol_11_nn_hpo.png", [
        "Подбор гиперпараметров самой НС через Optuna (TPE-сэмплер, seed=42).",
        "",
        f"Бюджет: {HPO_TRIALS} проб; целевая метрика - {HPO_CV_FOLDS}-fold CV accuracy на train",
        f"(укороченное обучение {HPO_TUNE_EPOCHS} эпох в поиске, без утечки из теста).",
        "Пространство поиска: hidden in {4,6,8,10,12}, learning_rate in [0.05, 0.9] (log).",
        "",
        f"Базовый конфиг ЛР-2:    hidden={NN_HIDDEN}, lr={NN_LEARNING_RATE}",
        f"Лучший конфиг Optuna:   hidden={best_params['hidden']}, "
        f"lr={best_params['learning_rate']:.4f}",
        f"Лучшая CV accuracy:     {study.best_value:.4f}",
        "",
        "Метрики на тестовой выборке (тот же сплит):",
        f"  базовая НС:  accuracy={nn_metrics['accuracy']:.4f}, AUC={nn_metrics['roc_auc']:.4f}, "
        f"F1={nn_metrics['f1']:.4f}",
        f"  НС+Optuna:   accuracy={tuned_metrics['accuracy']:.4f}, AUC={tuned_metrics['roc_auc']:.4f}, "
        f"F1={tuned_metrics['f1']:.4f}",
    ], font_size=10)

    print("\nГотово. Артефакты сохранены в lab8/artifacts, протоколы в lab8/protocol.")
    print(f"  H2O лидер: {h2o_result['leader_id']}")
    print(f"  FEDOT пайплайн: {' -> '.join(fedot_result['nodes'])}")


if __name__ == "__main__":
    main()

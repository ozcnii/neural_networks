import csv
import math
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


SEED = 42
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
DATASET_PATH = os.path.join(DATA_DIR, "student_dropout_risk_dataset.csv")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")
TRAINING_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.png")
PROTOCOL_HISTORY_PATH = os.path.join(PROTOCOL_DIR, "protocol_3_training_history.png")

FEATURE_COLUMNS = [
    "average_grade",
    "attendance_percent",
    "academic_debts",
    "study_hours_per_week",
    "class_activity_score",
]


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(activated_value):
    return activated_value * (1.0 - activated_value)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def generate_student_dataset(path, rows=240, seed=SEED):
    rng = random.Random(seed)
    records = []

    for _ in range(rows):
        average_grade = clamp(rng.gauss(3.75, 0.62), 2.0, 5.0)
        attendance_percent = clamp(rng.gauss(76.0, 16.0), 25.0, 100.0)
        academic_debts = int(clamp(round(rng.gauss(1.7, 1.6)), 0, 7))
        study_hours_per_week = clamp(rng.gauss(10.5, 4.5), 1.0, 28.0)
        class_activity_score = clamp(rng.gauss(6.0, 2.1), 0.0, 10.0)

        risk_score = (
            1.25 * academic_debts
            + 0.075 * (100.0 - attendance_percent)
            + 0.92 * (5.0 - average_grade)
            + 0.18 * (10.0 - class_activity_score)
            + 0.08 * max(0.0, 12.0 - study_hours_per_week)
            + rng.uniform(-0.8, 0.8)
        )
        high_risk = 1 if risk_score >= 6.0 else 0

        records.append(
            {
                "average_grade": round(average_grade, 2),
                "attendance_percent": round(attendance_percent, 1),
                "academic_debts": academic_debts,
                "study_hours_per_week": round(study_hours_per_week, 1),
                "class_activity_score": round(class_activity_score, 1),
                "high_dropout_risk": high_risk,
            }
        )

    rng.shuffle(records)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*FEATURE_COLUMNS, "high_dropout_risk"])
        writer.writeheader()
        writer.writerows(records)

    return records


def load_dataset(path):
    features = []
    targets = []

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            targets.append([float(row["high_dropout_risk"])])

    return np.array(features, dtype=float), np.array(targets, dtype=float)


def normalize_features(features):
    normalized = np.zeros_like(features, dtype=float)
    normalized[:, 0] = (features[:, 0] - 2.0) / 3.0
    normalized[:, 1] = features[:, 1] / 100.0
    normalized[:, 2] = features[:, 2] / 7.0
    normalized[:, 3] = features[:, 3] / 28.0
    normalized[:, 4] = features[:, 4] / 10.0
    return normalized


def split_dataset(features, targets, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]
    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
        train_indices,
        test_indices,
    )


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
    def __init__(self, input_count, layer_sizes, learning_rate=0.45, seed=SEED):
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

    def predict(self, inputs):
        return 1 if self.predict_probability(inputs) >= 0.5 else 0

    def fit(self, features, targets, max_epochs=5000, error_threshold=0.012):
        history = []

        for epoch in range(1, max_epochs + 1):
            mse = 0.0

            for inputs, target in zip(features, targets):
                output = self.forward(inputs)
                self.backward(inputs, target)
                mse += float(np.sum((target - output) ** 2))

            mse /= len(features)
            train_accuracy = self.accuracy(features, targets)
            history.append(
                {
                    "epoch": epoch,
                    "mse": mse,
                    "accuracy": train_accuracy,
                }
            )

            if epoch == 1 or epoch % 250 == 0 or mse < error_threshold:
                print(f"  Эпоха {epoch:>4d}: MSE = {mse:.6f}, Accuracy = {train_accuracy:.2%}")

            if mse < error_threshold:
                print(f"  Остановка: MSE {mse:.6f} < {error_threshold}")
                break

        return history

    def accuracy(self, features, targets):
        correct = 0
        for inputs, target in zip(features, targets):
            correct += int(self.predict(inputs) == int(target[0]))
        return correct / len(targets)


def class_distribution(records):
    low = sum(1 for row in records if row["high_dropout_risk"] == 0)
    high = sum(1 for row in records if row["high_dropout_risk"] == 1)
    return low, high


def save_training_log(history):
    with open(TRAINING_LOG_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "mse", "accuracy"])
        writer.writeheader()
        writer.writerows(history)


def save_training_plot(history):
    epochs = [row["epoch"] for row in history]
    losses = [row["mse"] for row in history]
    accuracies = [row["accuracy"] for row in history]

    plt.figure(figsize=(10, 5.6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, linewidth=2, label="MSE")
    plt.axhline(0.012, color="gray", linestyle="--", linewidth=1, label="Порог")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Ошибка обучения")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, linewidth=2, color="#2a7f62", label="Train accuracy")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.title("Точность на train")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=160)
    plt.savefig(PROTOCOL_HISTORY_PATH, dpi=160)
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


def format_feature_row(raw_features, target):
    return (
        f"grade={raw_features[0]:.2f}, attendance={raw_features[1]:5.1f}, "
        f"debts={int(raw_features[2])}, hours={raw_features[3]:4.1f}, "
        f"activity={raw_features[4]:4.1f} -> risk={int(target[0])}"
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROTOCOL_DIR, exist_ok=True)

    print("Лабораторная работа 2. Двухслойная нейронная сеть")
    print("Задача: бинарная классификация риска отчисления студента")
    print()

    records = generate_student_dataset(DATASET_PATH)
    raw_features, targets = load_dataset(DATASET_PATH)
    features = normalize_features(raw_features)
    low_count, high_count = class_distribution(records)

    print("1. Загрузка данных")
    print(f"CSV создан и загружен: {DATASET_PATH}")
    print(f"Количество строк: {len(records)}")
    print(f"Низкий риск: {low_count}, высокий риск: {high_count}")
    print()

    X_train, X_test, y_train, y_test, train_indices, test_indices = split_dataset(features, targets)

    print("2. Разбиение выборки")
    print(f"Train: {len(X_train)} строк")
    print(f"Test: {len(X_test)} строк")
    print()

    network = NeuralNetwork(input_count=len(FEATURE_COLUMNS), layer_sizes=[6, 1], learning_rate=0.45)

    print("3. Создание нейронной сети")
    print(f"Входной слой: {len(FEATURE_COLUMNS)} признаков")
    print("Скрытый слой: 6 нейронов, sigmoid")
    print("Выходной слой: 1 нейрон, sigmoid")
    print("Функция ошибки: MSE")
    print()

    print("4. Обучение")
    history = network.fit(X_train, y_train, max_epochs=5000, error_threshold=0.012)
    save_training_log(history)
    print(f"Лог обучения сохранен: {TRAINING_LOG_PATH}")
    print()

    train_accuracy = network.accuracy(X_train, y_train)
    test_accuracy = network.accuracy(X_test, y_test)

    print("5. Оценка")
    print(f"Точность на train: {train_accuracy:.2%}")
    print(f"Точность на test:  {test_accuracy:.2%}")
    print()

    print("6. Предсказания на тестовой выборке")
    prediction_lines = []
    for number, dataset_index in enumerate(test_indices[:10], start=1):
        normalized_inputs = features[dataset_index]
        probability = network.predict_probability(normalized_inputs)
        prediction = 1 if probability >= 0.5 else 0
        actual = int(targets[dataset_index][0])
        mark = "OK" if prediction == actual else "ERR"
        row = raw_features[dataset_index]
        line = (
            f"{number:>2}. {format_feature_row(row, targets[dataset_index])}; "
            f"pred={prediction}, p={probability:.3f}, {mark}"
        )
        print(line)
        prediction_lines.append(line)

    save_training_plot(history)

    final_epoch = history[-1]
    stop_reason = (
        f"MSE {final_epoch['mse']:.6f} < 0.012"
        if final_epoch["mse"] < 0.012
        else "достигнут лимит эпох"
    )

    first_rows = [
        format_feature_row(raw_features[index], targets[index])
        for index in range(min(18, len(raw_features)))
    ]

    save_text_protocol(
        "protocol_1_dataset.png",
        [
            f"Файл: {os.path.basename(DATASET_PATH)}",
            f"Количество записей: {len(records)}",
            f"Признаки: {', '.join(FEATURE_COLUMNS)}",
            f"Классы: 0 - низкий риск, 1 - высокий риск",
            f"Распределение: низкий риск={low_count}, высокий риск={high_count}",
            "",
            "Первые строки выборки:",
            *first_rows,
        ],
        font_size=10,
    )
    save_text_protocol(
        "protocol_2_network.png",
        [
            f"Вход: {len(FEATURE_COLUMNS)} нормализованных признаков",
            "Скрытый слой: 6 нейронов, функция активации sigmoid",
            "Выходной слой: 1 нейрон, функция активации sigmoid",
            "Правило класса: probability >= 0.5 -> высокий риск",
            "Функция ошибки: MSE",
            "Алгоритм обучения: обратное распространение ошибки",
            "Обновление весов выполняется после каждого объекта train-выборки.",
        ],
    )
    save_text_protocol(
        "protocol_4_training_result.png",
        [
            f"Train: {len(X_train)} строк, Test: {len(X_test)} строк",
            f"Остановка: {stop_reason}",
            f"Итоговая эпоха: {int(final_epoch['epoch'])}",
            f"Итоговый MSE: {final_epoch['mse']:.6f}",
            f"Train accuracy: {train_accuracy:.2%}",
            f"Test accuracy:  {test_accuracy:.2%}",
            f"Лог обучения: {os.path.basename(TRAINING_LOG_PATH)}",
            f"График обучения: {os.path.basename(TRAINING_HISTORY_PATH)}",
        ],
    )
    save_text_protocol(
        "protocol_5_predictions.png",
        [
            "risk - ожидаемый класс, pred - предсказанный класс, p - вероятность высокого риска",
            "",
            *prediction_lines,
        ],
        font_size=9,
    )

    print()
    print("Артефакты обновлены:")
    for artifact in [
        DATASET_PATH,
        TRAINING_LOG_PATH,
        TRAINING_HISTORY_PATH,
        PROTOCOL_HISTORY_PATH,
        os.path.join(PROTOCOL_DIR, "protocol_1_dataset.png"),
        os.path.join(PROTOCOL_DIR, "protocol_2_network.png"),
        os.path.join(PROTOCOL_DIR, "protocol_4_training_result.png"),
        os.path.join(PROTOCOL_DIR, "protocol_5_predictions.png"),
    ]:
        print(f"  {artifact}")


if __name__ == "__main__":
    main()

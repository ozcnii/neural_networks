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
import pandas as pd


SEED = 42
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
DATASET_PATH = os.path.join(DATA_DIR, "student_performance_dataset.csv")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")
TRAINING_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.png")
PROTOCOL_HISTORY_PATH = os.path.join(PROTOCOL_DIR, "protocol_6_training_history.png")

FEATURE_COLUMNS = [
    "attendance_percent",
    "homework_completion_percent",
    "average_test_score",
    "study_hours_per_week",
    "sleep_hours",
    "stress_level",
    "missed_deadlines",
]

label_mapping = {
    "усилить подготовку": 0,
    "держать темп": 1,
    "готов к экзамену": 2,
}
reverse_label_mapping = {index: label for label, index in label_mapping.items()}


def relu(value):
    return max(0.0, value)


def relu_derivative(value):
    return 1.0 if value > 0.0 else 0.0


def sigmoid(value):
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def sigmoid_derivative(value):
    activated = sigmoid(value)
    return activated * (1.0 - activated)


def tanh(value):
    return math.tanh(value)


def tanh_derivative(value):
    activated = math.tanh(value)
    return 1.0 - activated * activated


def softmax(values):
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def activation_forward(value, activation):
    if activation == "relu":
        return relu(value)
    if activation == "sigmoid":
        return sigmoid(value)
    if activation == "tanh":
        return tanh(value)
    return value


def activation_derivative(value, activation):
    if activation == "relu":
        return relu_derivative(value)
    if activation == "sigmoid":
        return sigmoid_derivative(value)
    if activation == "tanh":
        return tanh_derivative(value)
    return 1.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def class_from_readiness(readiness_score):
    if readiness_score < 40.0:
        return "усилить подготовку"
    if readiness_score < 60.0:
        return "держать темп"
    return "готов к экзамену"


def generate_student_dataset(path, rows=540, seed=SEED):
    rng = random.Random(seed)
    records = []

    for _ in range(rows):
        attendance = clamp(rng.gauss(78.0, 14.0), 35.0, 100.0)
        homework = clamp(rng.gauss(73.0, 18.0), 20.0, 100.0)
        average_test_score = clamp(rng.gauss(69.0, 16.0), 25.0, 100.0)
        study_hours = clamp(rng.gauss(11.0, 5.0), 1.0, 28.0)
        sleep_hours = clamp(rng.gauss(7.0, 1.1), 4.5, 9.5)
        stress_level = clamp(rng.gauss(5.2, 2.2), 0.0, 10.0)
        missed_deadlines = int(clamp(round(rng.gauss(2.0, 1.6)), 0, 8))

        readiness_score = (
            0.24 * attendance
            + 0.23 * homework
            + 0.31 * average_test_score
            + 1.55 * study_hours
            + 2.10 * sleep_hours
            - 2.35 * stress_level
            - 3.70 * missed_deadlines
            + rng.uniform(-4.0, 4.0)
            - 19.0
        )

        records.append(
            {
                "attendance_percent": round(attendance, 1),
                "homework_completion_percent": round(homework, 1),
                "average_test_score": round(average_test_score, 1),
                "study_hours_per_week": round(study_hours, 1),
                "sleep_hours": round(sleep_hours, 1),
                "stress_level": round(stress_level, 1),
                "missed_deadlines": missed_deadlines,
                "recommended_action": class_from_readiness(readiness_score),
            }
        )

    dataframe = pd.DataFrame(records)
    dataframe = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dataframe.to_csv(path, index=False)
    return dataframe


def load_dataset(path):
    dataframe = pd.read_csv(path)
    features = dataframe[FEATURE_COLUMNS].astype(float)
    labels = dataframe["recommended_action"].map(label_mapping).astype(int)
    return dataframe, features, labels


def split_dataset(features, labels, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    train_end = int(len(indices) * 0.70)
    validation_end = train_end + int(len(indices) * 0.15)

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    return train_indices, validation_indices, test_indices


def normalize_by_train(features, train_indices):
    train_features = features.iloc[train_indices]
    minimums = train_features.min()
    maximums = train_features.max()
    scale = (maximums - minimums).replace(0, 1)
    normalized = (features - minimums) / scale
    return normalized.to_numpy(dtype=float), minimums, maximums


class Neuron:
    def __init__(self, input_size, activation="linear", rng=None):
        self.activation = activation
        self.rng = rng or np.random.default_rng(SEED)
        limit = math.sqrt(6.0 / (input_size + 1))
        self.weights = self.rng.uniform(-limit, limit, input_size)
        self.bias = 0.0
        self.last_inputs = None
        self.last_z = 0.0
        self.weight_gradients = np.zeros(input_size)
        self.bias_gradient = 0.0

    def forward(self, inputs):
        self.last_inputs = np.array(inputs, dtype=float)
        self.last_z = float(np.dot(self.weights, self.last_inputs) + self.bias)
        return activation_forward(self.last_z, self.activation)

    def backward(self, output_gradient):
        local_gradient = output_gradient * activation_derivative(self.last_z, self.activation)
        self.weight_gradients += local_gradient * self.last_inputs
        self.bias_gradient += local_gradient
        return local_gradient * self.weights

    def update(self, learning_rate, batch_size):
        self.weights -= learning_rate * self.weight_gradients / batch_size
        self.bias -= learning_rate * self.bias_gradient / batch_size
        self.weight_gradients.fill(0.0)
        self.bias_gradient = 0.0


class DenseLayer:
    def __init__(self, input_size, output_size, activation="linear", rng=None):
        self.neurons = [
            Neuron(input_size, activation=activation, rng=rng)
            for _ in range(output_size)
        ]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons], dtype=float)

    def backward(self, output_gradients):
        input_gradients = np.zeros_like(self.neurons[0].weights)
        for neuron, gradient in zip(self.neurons, output_gradients):
            input_gradients += neuron.backward(float(gradient))
        return input_gradients

    def update(self, learning_rate, batch_size):
        for neuron in self.neurons:
            neuron.update(learning_rate, batch_size)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        outputs = np.array(inputs, dtype=float)
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def predict(self, inputs):
        logits = self.forward(inputs)
        probabilities = softmax(logits)
        return int(np.argmax(probabilities)), probabilities

    def compute_loss(self, probabilities, target):
        return -math.log(max(float(probabilities[target]), 1e-12))

    def compute_gradients(self, probabilities, target):
        gradients = probabilities.copy()
        gradients[target] -= 1.0
        return gradients

    def train_on_batch(self, batch_features, batch_targets, learning_rate):
        batch_loss = 0.0
        batch_correct = 0

        for inputs, target in zip(batch_features, batch_targets):
            logits = self.forward(inputs)
            probabilities = softmax(logits)
            prediction = int(np.argmax(probabilities))
            batch_correct += int(prediction == target)
            batch_loss += self.compute_loss(probabilities, target)

            gradients = self.compute_gradients(probabilities, target)
            for layer in reversed(self.layers):
                gradients = layer.backward(gradients)

        for layer in self.layers:
            layer.update(learning_rate, len(batch_features))

        return batch_loss / len(batch_features), batch_correct / len(batch_features)

    def evaluate(self, features, targets):
        total_loss = 0.0
        correct = 0

        for inputs, target in zip(features, targets):
            prediction, probabilities = self.predict(inputs)
            total_loss += self.compute_loss(probabilities, target)
            correct += int(prediction == target)

        return total_loss / len(features), correct / len(features)

    def train(
        self,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        epochs=100,
        batch_size=32,
        learning_rate=0.08,
        decay=0.985,
        target_val_loss=0.35,
    ):
        rng = np.random.default_rng(SEED)
        history = []

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(len(train_features))
            current_learning_rate = learning_rate * (decay ** (epoch - 1))
            batch_losses = []
            batch_accuracies = []

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_loss, batch_accuracy = self.train_on_batch(
                    train_features[batch_indices],
                    train_targets[batch_indices],
                    current_learning_rate,
                )
                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_accuracy)

            train_loss = float(np.mean(batch_losses))
            train_accuracy = float(np.mean(batch_accuracies))
            validation_loss, validation_accuracy = self.evaluate(
                validation_features,
                validation_targets,
            )

            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": current_learning_rate,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": validation_loss,
                    "val_accuracy": validation_accuracy,
                }
            )

            if epoch == 1 or epoch % 10 == 0 or validation_loss < target_val_loss:
                print(
                    f"Epoch {epoch:03d}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
                    f"Val Loss={validation_loss:.4f}, Val Acc={validation_accuracy:.4f}, "
                    f"LR={current_learning_rate:.4f}"
                )

            if validation_loss < target_val_loss:
                print(f"Early stopping: Val Loss {validation_loss:.4f} < {target_val_loss:.2f}")
                break

        return history


def save_training_plot(history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    plt.figure(figsize=(10, 5.6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.axhline(0.35, color="gray", linestyle="--", linewidth=1, label="Target Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ошибка обучения")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_accuracy, label="Val Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Точность классификации")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH, dpi=160)
    plt.savefig(PROTOCOL_HISTORY_PATH, dpi=160)
    plt.close()


def save_text_protocol(filename, _title, lines):
    path = os.path.join(PROTOCOL_DIR, filename)
    height = max(3.0, 0.55 + 0.34 * len(lines))
    plt.figure(figsize=(10, height))
    plt.axis("off")
    plt.text(
        0.02,
        0.95,
        "\n".join(lines),
        fontsize=11,
        family="monospace",
        va="top",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def format_mapping():
    return [f"{index}: {label}" for label, index in label_mapping.items()]


def class_distribution(series):
    counts = series.value_counts().reindex(label_mapping.keys(), fill_value=0)
    return [f"{label}: {int(count)}" for label, count in counts.items()]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROTOCOL_DIR, exist_ok=True)

    print("Лабораторная работа 1. Нейросеть своими руками")
    print("Задача: классификация рекомендации по подготовке студента к экзамену")
    print()

    dataframe = generate_student_dataset(DATASET_PATH)
    loaded_dataframe, features, labels = load_dataset(DATASET_PATH)

    print("1. Загрузка данных")
    print(f"CSV создан и загружен: {DATASET_PATH}")
    print(f"Количество строк: {len(loaded_dataframe)}")
    print(f"Признаки: {', '.join(FEATURE_COLUMNS)}")
    print("Метки классов:")
    for item in format_mapping():
        print(f"  {item}")
    print("Распределение классов:")
    for item in class_distribution(loaded_dataframe["recommended_action"]):
        print(f"  {item}")
    print()

    train_indices, validation_indices, test_indices = split_dataset(features, labels)
    normalized_features, minimums, maximums = normalize_by_train(features, train_indices)
    targets = labels.to_numpy(dtype=int)

    train_features = normalized_features[train_indices]
    validation_features = normalized_features[validation_indices]
    test_features = normalized_features[test_indices]
    train_targets = targets[train_indices]
    validation_targets = targets[validation_indices]
    test_targets = targets[test_indices]

    print("2. Разбиение выборки")
    print(f"Train: {len(train_features)} строк")
    print(f"Validation: {len(validation_features)} строк")
    print(f"Test: {len(test_features)} строк")
    print("Нормализация min-max рассчитана только по train-части")
    print()

    rng = np.random.default_rng(SEED)
    network = NeuralNetwork()
    network.add_layer(DenseLayer(len(FEATURE_COLUMNS), 14, activation="relu", rng=rng))
    network.add_layer(DenseLayer(14, len(label_mapping), activation="linear", rng=rng))

    print("3. Создание нейронной сети")
    print(f"Входной слой: {len(FEATURE_COLUMNS)} признаков")
    print("Скрытый слой: 14 нейронов, ReLU")
    print(f"Выходной слой: {len(label_mapping)} нейрона, Softmax")
    print("Функция ошибки: cross-entropy")
    print()

    print("4. Обучение")
    history = network.train(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        epochs=100,
        batch_size=32,
        learning_rate=0.08,
        decay=0.985,
        target_val_loss=0.35,
    )
    pd.DataFrame(history).to_csv(TRAINING_LOG_PATH, index=False)
    print(f"Лог обучения сохранен: {TRAINING_LOG_PATH}")
    print()

    test_loss, test_accuracy = network.evaluate(test_features, test_targets)
    print("5. Оценка на тестовых данных")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print()

    save_training_plot(history)
    print("6. График обучения")
    print(f"График сохранен: {TRAINING_HISTORY_PATH}")
    print()

    print("7. Примеры предсказаний")
    prediction_lines = []
    for number, index in enumerate(test_indices[:5], start=1):
        inputs = normalized_features[index]
        prediction, probabilities = network.predict(inputs)
        actual = targets[index]
        row = dataframe.iloc[index]
        probability_text = ", ".join(
            f"{reverse_label_mapping[class_index]}={probability:.2f}"
            for class_index, probability in enumerate(probabilities)
        )
        line = (
            f"{number}. факт: {reverse_label_mapping[actual]}; "
            f"прогноз: {reverse_label_mapping[prediction]}; "
            f"вероятности: {probability_text}"
        )
        print(line)
        prediction_lines.append(
            f"{number}. явка={row['attendance_percent']:.1f}, дз={row['homework_completion_percent']:.1f}, "
            f"тест={row['average_test_score']:.1f}, часы={row['study_hours_per_week']:.1f} -> "
            f"факт: {reverse_label_mapping[actual]}, прогноз: {reverse_label_mapping[prediction]}"
        )
        prediction_lines.append(f"   {probability_text}")

    final_epoch = history[-1]
    stop_reason = (
        f"Val Loss {final_epoch['val_loss']:.4f} < 0.35"
        if final_epoch["val_loss"] < 0.35
        else "достигнут лимит эпох"
    )

    save_text_protocol(
        "protocol_1_loading.png",
        "1. Загрузка и генерация данных",
        [
            f"CSV: {os.path.basename(DATASET_PATH)}",
            f"Количество записей: {len(loaded_dataframe)}",
            f"Количество признаков: {len(FEATURE_COLUMNS)}",
            "Признаки:",
            *[f"  - {column}" for column in FEATURE_COLUMNS],
            "Классы:",
            *[f"  - {item}" for item in format_mapping()],
            "Распределение:",
            *[f"  - {item}" for item in class_distribution(loaded_dataframe["recommended_action"])],
        ],
    )
    save_text_protocol(
        "protocol_2_split.png",
        "2. Разбиение и нормализация",
        [
            f"Train: {len(train_features)} строк (70%)",
            f"Validation: {len(validation_features)} строк (15%)",
            f"Test: {len(test_features)} строк (15%)",
            "Min-max нормализация рассчитана по train-выборке.",
            "Диапазоны train:",
            *[
                f"  - {column}: {minimums[column]:.2f} .. {maximums[column]:.2f}"
                for column in FEATURE_COLUMNS
            ],
        ],
    )
    save_text_protocol(
        "protocol_3_network.png",
        "3. Создание нейронной сети",
        [
            f"Вход: {len(FEATURE_COLUMNS)} числовых признаков",
            "DenseLayer 1: 14 нейронов, activation=relu",
            f"DenseLayer 2: {len(label_mapping)} нейрона, activation=linear",
            "Вероятности классов вычисляются функцией softmax.",
            "Ошибка: categorical cross-entropy.",
            "Обратное распространение ошибки реализовано вручную.",
        ],
    )
    save_text_protocol(
        "protocol_4_training.png",
        "4. Обучение сети",
        [
            "Параметры: epochs=100, batch_size=32, learning_rate=0.08, decay=0.985",
            f"Остановка: {stop_reason}",
            f"Итоговая эпоха: {int(final_epoch['epoch'])}",
            f"Train Loss: {final_epoch['train_loss']:.4f}",
            f"Train Accuracy: {final_epoch['train_accuracy']:.4f}",
            f"Val Loss: {final_epoch['val_loss']:.4f}",
            f"Val Accuracy: {final_epoch['val_accuracy']:.4f}",
            f"Лог сохранен: {os.path.basename(TRAINING_LOG_PATH)}",
        ],
    )
    save_text_protocol(
        "protocol_5_test.png",
        "5. Оценка на тестовой выборке",
        [
            f"Test rows: {len(test_features)}",
            f"Test Loss: {test_loss:.4f}",
            f"Test Accuracy: {test_accuracy:.4f}",
            "Классификация выполняется по максимальной вероятности softmax.",
        ],
    )
    save_text_protocol(
        "protocol_7_predictions.png",
        "7. Примеры предсказаний",
        prediction_lines,
    )

    print()
    print("Артефакты обновлены:")
    for artifact in [
        DATASET_PATH,
        TRAINING_LOG_PATH,
        TRAINING_HISTORY_PATH,
        PROTOCOL_HISTORY_PATH,
        os.path.join(PROTOCOL_DIR, "protocol_1_loading.png"),
        os.path.join(PROTOCOL_DIR, "protocol_2_split.png"),
        os.path.join(PROTOCOL_DIR, "protocol_3_network.png"),
        os.path.join(PROTOCOL_DIR, "protocol_4_training.png"),
        os.path.join(PROTOCOL_DIR, "protocol_5_test.png"),
        os.path.join(PROTOCOL_DIR, "protocol_7_predictions.png"),
    ]:
        print(f"  {artifact}")


if __name__ == "__main__":
    main()

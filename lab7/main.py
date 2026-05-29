import copy
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

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Воспроизводимость
# =========================
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROTOCOL_DIR = os.path.join(BASE_DIR, "protocol")
DATA_DIR = os.path.join(BASE_DIR, "data")
for d in (ARTIFACTS_DIR, PROTOCOL_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
TRAINING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")
TUNING_LOG_PATH = os.path.join(ARTIFACTS_DIR, "tuning_log.csv")
LOSS_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "loss_curve.png")
PERPLEXITY_PATH = os.path.join(ARTIFACTS_DIR, "perplexity.png")
ATTENTION_PATH = os.path.join(ARTIFACTS_DIR, "attention.png")
SAMPLES_PATH = os.path.join(ARTIFACTS_DIR, "samples.txt")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Гиперпараметры
# =========================
BLOCK_SIZE = 96           # длина контекста (символов), которую видит модель
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 3e-3
WEIGHT_DECAY = 0.1
TRAIN_FRAC = 0.90         # первые 90% текста — train, последние 10% — val

EVAL_INTERVAL = 250       # как часто оценивать loss во время обучения
EVAL_ITERS = 25           # число батчей для усреднённой оценки loss

# Этап подбора конфигурации: короткие прогоны-кандидаты для честного выбора.
SCREEN_ITERS = 500
SCREEN_CONFIGS = [
    {"name": "mini", "n_layer": 2, "n_head": 4, "n_embd": 96},
    {"name": "base", "n_layer": 3, "n_head": 4, "n_embd": 128},
    {"name": "wide", "n_layer": 4, "n_head": 4, "n_embd": 128},
]
# Финальное обучение выбранной конфигурации.
FINAL_ITERS = 2200

# Параметры авторегрессионной генерации (температура + top-k).
GEN_TOKENS = 500
GEN_SETTINGS = [
    {"temperature": 0.5, "top_k": 20},
    {"temperature": 0.8, "top_k": 40},
    {"temperature": 1.0, "top_k": 0},
]
GEN_PROMPT = "\n"


# =========================
# Корпус и char-level токенизация
# =========================
def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return text, chars, stoi, itos, encode, decode


def build_splits(text, encode):
    """Разбиение по тексту без перемешивания: в каждой сказке первые TRAIN_FRAC
    идут в train, последний хвост — в val. Так val представляет все сказки
    (а не только хвост последней), оставаясь непрерывными фрагментами."""
    tales = [t for t in text.split("\n\n\n") if t.strip()]
    train_parts, val_parts = [], []
    for tale in tales:
        cut = int(len(tale) * TRAIN_FRAC)
        train_parts.append(tale[:cut])
        val_parts.append(tale[cut:])
    train_data = torch.tensor(encode("".join(train_parts)), dtype=torch.long)
    val_data = torch.tensor(encode("".join(val_parts)), dtype=torch.long)
    return tales, train_data, val_data


def get_batch(data, block_size, batch_size, generator):
    """Случайные окна: вход x=[i:i+T], цель y=[i+1:i+T+1] (сдвиг на 1 символ)."""
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# =========================
# Самописный причинный multi-head self-attention
# =========================
class CausalSelfAttention(nn.Module):
    """Self-attention с нуля: Q/K/V на nn.Linear, причинная маска и softmax вручную."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # нижнетреугольная маска: позиция t видит только токены <= t
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("tril", mask)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # масштабированное скалярное произведение: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # причинная маска: будущее заполняем -inf, чтобы softmax дал там 0
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                   # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        if return_attn:
            return y, att
        return y


class MLP(nn.Module):
    """Позиционно-независимый блок: Linear -> GELU -> Linear (расширение 4x)."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Блок трансформера: pre-LN -> attention -> residual; pre-LN -> MLP -> residual."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, return_attn=False):
        if return_attn:
            a, att = self.attn(self.ln1(x), return_attn=True)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, att
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Самописный GPT: эмбеддинги + стек блоков + LN + линейная голова (weight tying)."""

    def __init__(self, vocab_size, n_layer, n_head, n_embd,
                 block_size=BLOCK_SIZE, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight        # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def collect_attention(self, idx):
        """Прогон с захватом карт внимания каждого блока (для визуализации)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        maps = []
        for block in self.blocks:
            x, att = block(x, return_attn=True)
            maps.append(att)
        return maps

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k, generator):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def count_params(model):
    # при weight tying голова делит веса с эмбеддингом — считаем уникальные тензоры
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


# =========================
# Оценка loss и обучение
# =========================
@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size):
    """Усреднённый loss по случайным батчам (для кривой обучения)."""
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        gen = torch.Generator()
        gen.manual_seed(SEED + (0 if name == "train" else 1))
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(data, block_size, BATCH_SIZE, gen)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def full_pass_loss(model, data, block_size):
    """Детерминированный per-char loss по всей выборке (непересекающиеся окна)."""
    model.eval()
    n = (len(data) - 1) // block_size
    total, count = 0.0, 0
    for i in range(n):
        s = i * block_size
        x = data[s:s + block_size].unsqueeze(0).to(device)
        y = data[s + 1:s + block_size + 1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        total += loss.item() * block_size
        count += block_size
    return total / count


def train_model(config, train_data, val_data, vocab_size, max_iters, log=False):
    """Обучение одной конфигурации с сохранением лучшего по val чекпойнта."""
    set_seed(SEED)
    model = GPT(vocab_size, config["n_layer"], config["n_head"],
                config["n_embd"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    gen = torch.Generator()
    gen.manual_seed(SEED)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_iter = 0
    log_rows = []

    model.train()
    for it in range(1, max_iters + 1):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, gen)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == max_iters:
            stats = estimate_loss(model, train_data, val_data, BLOCK_SIZE)
            log_rows.append((it, stats["train"], stats["val"]))
            improved = stats["val"] < best_val - 1e-4
            if improved:
                best_val = stats["val"]
                best_state = copy.deepcopy(model.state_dict())
                best_iter = it
            if log:
                print(
                    f"шаг {it:>4d}/{max_iters} | "
                    f"train loss={stats['train']:.4f} | val loss={stats['val']:.4f}"
                    f"{'  <- лучший' if improved else ''}"
                )

    model.load_state_dict(best_state)
    return model, log_rows, best_iter, best_val


# =========================
# Статистические бейзлайны (unigram, bigram) на char-level
# =========================
def baseline_perplexities(train_data, val_data, vocab_size):
    """Перплексия наивных моделей на val: unigram и bigram (сглаживание +1)."""
    train = train_data.numpy()
    val = val_data.numpy()

    # unigram: P(c) по частотам train
    uni = np.bincount(train, minlength=vocab_size).astype(np.float64) + 1.0
    uni /= uni.sum()
    uni_nll = -np.mean(np.log(uni[val]))

    # bigram: P(c_t | c_{t-1}) по парам train
    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)  # +1 сглаживание
    np.add.at(counts, (train[:-1], train[1:]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    bi_nll = -np.mean(np.log(probs[val[:-1], val[1:]]))

    return {
        "unigram": math.exp(uni_nll),
        "bigram": math.exp(bi_nll),
    }


# =========================
# Визуализация
# =========================
def save_text_protocol(filename, lines, font_size=11):
    path = os.path.join(PROTOCOL_DIR, filename)
    height = max(3.2, 0.6 + 0.33 * len(lines))
    plt.figure(figsize=(10, height))
    plt.axis("off")
    plt.text(0.02, 0.98, "\n".join(lines), fontsize=font_size,
             family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_loss_curve(log_rows, best_iter):
    iters = [r[0] for r in log_rows]
    train_loss = [r[1] for r in log_rows]
    val_loss = [r[2] for r in log_rows]

    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    ax1.plot(iters, train_loss, label="train loss", color="#2980b9", linewidth=2)
    ax1.plot(iters, val_loss, label="val loss", color="#c0392b", linewidth=2)
    ax1.axvline(best_iter, color="#27ae60", linestyle="--", linewidth=1.5,
                label=f"лучший чекпойнт (шаг {best_iter})")
    ax1.set_xlabel("Шаг обучения")
    ax1.set_ylabel("Кросс-энтропия (loss)")
    ax1.grid(True, alpha=0.3)

    # вторая ось: перплексия = exp(loss) — нагляднее для оценки качества
    ax2 = ax1.twinx()
    lo, hi = ax1.get_ylim()
    ax2.set_ylim(math.exp(lo), math.exp(hi))
    ax2.set_ylabel("Перплексия = exp(loss)")

    ax1.set_title("Кривая обучения самописного GPT (loss и перплексия)")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(LOSS_CURVE_PATH, dpi=160)
    fig.savefig(os.path.join(PROTOCOL_DIR, "protocol_5_loss_curve.png"), dpi=160)
    plt.close(fig)


def save_perplexity_plot(gpt_ppl, baselines):
    names = ["unigram", "bigram", "GPT"]
    values = [baselines["unigram"], baselines["bigram"], gpt_ppl]
    colors = ["#95a5a6", "#e67e22", "#27ae60"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors, width=0.6)
    top = max(values)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + top * 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    plt.ylabel("Перплексия на val (меньше — лучше)")
    plt.ylim(0, top * 1.15)
    plt.title("Перплексия на val: GPT против статистических бейзлайнов")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PERPLEXITY_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_6_perplexity.png"), dpi=160)
    plt.close()


def save_attention_heatmap(model, encode, sample_text):
    """Тепловая карта внимания одной головы последнего блока на коротком примере."""
    ids = torch.tensor([encode(sample_text)], dtype=torch.long, device=device)
    maps = model.collect_attention(ids)
    att = maps[-1][0, 0].detach().cpu().numpy()       # последний блок, голова 0
    labels = [c if c != "\n" else "\\n" for c in sample_text]

    plt.figure(figsize=(8.5, 7))
    plt.imshow(att, cmap="viridis", aspect="auto")
    plt.colorbar(label="вес внимания")
    plt.xticks(range(len(labels)), labels, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Ключи (на какие символы смотрит)")
    plt.ylabel("Запросы (текущий символ)")
    plt.title("Карта внимания: последний блок, голова 0\n"
              "(нижнетреугольная — будущее замаскировано)")
    plt.tight_layout()
    plt.savefig(ATTENTION_PATH, dpi=160)
    plt.savefig(os.path.join(PROTOCOL_DIR, "protocol_8_attention.png"), dpi=160)
    plt.close()


def wrap_sample(text, width=64, max_lines=12):
    """Готовит сгенерированный текст к показу на PNG: перенос строк и обрезка."""
    out = []
    for raw in text.split("\n"):
        if raw == "":
            out.append("")
            continue
        while len(raw) > width:
            out.append(raw[:width])
            raw = raw[width:]
        out.append(raw)
    if len(out) > max_lines:
        out = out[:max_lines] + ["..."]
    return out


# =========================
# main
# =========================
def main():
    text, chars, stoi, itos, encode, decode = load_corpus()
    tales, train_data, val_data = build_splits(text, encode)
    vocab_size = len(chars)

    print("=== ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")
    print("Задача: char-level генерация текста самописным GPT-трансформером")
    print("Корпус: сказки А. С. Пушкина (public domain)")
    print(f"Размер корпуса: {len(text)} символов ({len(text.encode('utf-8'))} байт)")
    print(f"Сказок в корпусе: {len(tales)}")
    print(f"Размер словаря: {vocab_size} уникальных символов")
    print(f"Контекст (block_size): {BLOCK_SIZE} символов")
    print(f"Разбиение по каждой сказке {int(TRAIN_FRAC*100)}/"
          f"{100-int(TRAIN_FRAC*100)} (без перемешивания): "
          f"train={len(train_data)}, val={len(val_data)} символов")
    print()

    # ---- Этап 1: подбор конфигурации короткими прогонами ----
    print("=== ПОДБОР КОНФИГУРАЦИИ (короткие прогоны) ===")
    tuning_rows = []
    for cfg in SCREEN_CONFIGS:
        model, _, _, best_val = train_model(
            cfg, train_data, val_data, vocab_size, SCREEN_ITERS, log=False
        )
        params = count_params(model)
        val_ppl = math.exp(full_pass_loss(model, val_data, BLOCK_SIZE))
        tuning_rows.append({
            "name": cfg["name"], "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"], "n_embd": cfg["n_embd"],
            "params": params, "val_loss": best_val, "val_ppl": val_ppl,
        })
        print(f"  {cfg['name']:<5} L={cfg['n_layer']} H={cfg['n_head']} "
              f"E={cfg['n_embd']:>3} | параметров={params:>7} | "
              f"val loss={best_val:.4f} | val ppl={val_ppl:.2f}")

    best_cfg_row = min(tuning_rows, key=lambda r: r["val_ppl"])
    final_cfg = next(c for c in SCREEN_CONFIGS if c["name"] == best_cfg_row["name"])
    pd.DataFrame(tuning_rows).to_csv(TUNING_LOG_PATH, index=False)
    print(f"Выбрана конфигурация '{final_cfg['name']}' "
          f"(минимальная val-перплексия за {SCREEN_ITERS} шагов).")
    print()

    # ---- Этап 2: полное обучение выбранной конфигурации ----
    print(f"=== ОБУЧЕНИЕ ВЫБРАННОЙ МОДЕЛИ ({final_cfg['name']}, "
          f"{FINAL_ITERS} шагов) ===")
    model, log_rows, best_iter, best_val = train_model(
        final_cfg, train_data, val_data, vocab_size, FINAL_ITERS, log=True
    )
    total_params = count_params(model)
    pd.DataFrame(log_rows, columns=["iter", "train_loss", "val_loss"]).to_csv(
        TRAINING_LOG_PATH, index=False
    )
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nЛучший чекпойнт: шаг {best_iter}, val loss={best_val:.4f}")
    print()

    # ---- Метрики: детерминированная перплексия train/val ----
    train_loss_full = full_pass_loss(model, train_data, BLOCK_SIZE)
    val_loss_full = full_pass_loss(model, val_data, BLOCK_SIZE)
    train_ppl = math.exp(train_loss_full)
    val_ppl = math.exp(val_loss_full)

    print("=== ИТОГОВЫЕ МЕТРИКИ (детерминированный проход) ===")
    print(f"train: loss={train_loss_full:.4f}, perplexity={train_ppl:.2f}")
    print(f"val:   loss={val_loss_full:.4f}, perplexity={val_ppl:.2f}")
    print(f"Параметров в модели: {total_params}")
    print()

    # ---- Бейзлайны ----
    baselines = baseline_perplexities(train_data, val_data, vocab_size)
    print("=== СРАВНЕНИЕ С БЕЙЗЛАЙНАМИ (перплексия на val) ===")
    print(f"  {'модель':<12}{'перплексия':>12}")
    print(f"  {'unigram':<12}{baselines['unigram']:>12.2f}")
    print(f"  {'bigram':<12}{baselines['bigram']:>12.2f}")
    print(f"  {'GPT':<12}{val_ppl:>12.2f}")
    gain = baselines["bigram"] / val_ppl
    print(f"GPT снижает перплексию в {gain:.1f} раза относительно биграммной модели.")
    print()

    # ---- Генерация ----
    print("=== ГЕНЕРАЦИЯ ТЕКСТА ===")
    samples = []
    for s in GEN_SETTINGS:
        gen = torch.Generator()
        gen.manual_seed(SEED)
        ctx = torch.tensor([encode(GEN_PROMPT)], dtype=torch.long, device=device)
        out = model.generate(ctx, GEN_TOKENS, s["temperature"], s["top_k"], gen)
        sample = decode(out[0].tolist())
        samples.append((s, sample))
        topk_str = s["top_k"] if s["top_k"] else "off"
        print(f"--- temperature={s['temperature']}, top_k={topk_str} ---")
        print(sample.strip()[:300])
        print()

    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        for s, sample in samples:
            topk_str = s["top_k"] if s["top_k"] else "off"
            f.write(f"=== temperature={s['temperature']}, top_k={topk_str} ===\n")
            f.write(sample.strip() + "\n\n")

    # ---- Графики ----
    save_loss_curve(log_rows, best_iter)
    save_perplexity_plot(val_ppl, baselines)
    attn_example = "Жил старик со своею старухой\n"
    save_attention_heatmap(model, encode, attn_example)

    # ---- Текстовые протоколы ----
    save_text_protocol(
        "protocol_1_dataset.png",
        [
            "Задача: char-level генерация текста самописным GPT-трансформером",
            "Корпус: сказки А. С. Пушкина (public domain)",
            "  - Сказка о царе Салтане",
            "  - Сказка о рыбаке и рыбке",
            "  - Сказка о мёртвой царевне и о семи богатырях",
            "  - Сказка о золотом петушке",
            "",
            f"Размер корпуса: {len(text)} символов "
            f"({len(text.encode('utf-8'))} байт)",
            f"Размер словаря: {vocab_size} уникальных символов",
            f"Длина контекста (block_size): {BLOCK_SIZE} символов",
            "Токенизация: char-level (char -> id), цель — следующий символ",
            f"Разбиение по {len(tales)} сказкам (без перемешивания): в каждой",
            f"  первые {int(TRAIN_FRAC*100)}% -> train, последние "
            f"{100-int(TRAIN_FRAC*100)}% -> val",
            f"  train = {len(train_data)} символов, val = {len(val_data)} символов",
        ],
        font_size=10,
    )

    arch_lines = [
        "Самописный GPT-трансформер (PyTorch, attention с нуля)",
        "",
        f"Выбранная конфигурация: '{final_cfg['name']}'",
        f"  n_layer = {final_cfg['n_layer']} (блоков трансформера)",
        f"  n_head  = {final_cfg['n_head']} (голов внимания)",
        f"  n_embd  = {final_cfg['n_embd']} (размерность эмбеддинга)",
        f"  block_size = {BLOCK_SIZE}, dropout = {DROPOUT}",
        "",
        "Структура блока:",
        "  pre-LayerNorm -> CausalSelfAttention (Q/K/V вручную) -> residual",
        "  pre-LayerNorm -> MLP (Linear->GELU->Linear, 4x) -> residual",
        "Эмбеддинги: токенные + позиционные; финальный LayerNorm;",
        "линейная голова в логиты по словарю (weight tying с эмбеддингом).",
        "",
        f"Всего обучаемых параметров: {total_params}",
        "Функция потерь: кросс-энтропия по символам",
        f"Оптимизатор: AdamW, lr={LR}, weight_decay={WEIGHT_DECAY}",
        f"Batch size: {BATCH_SIZE}, шагов обучения: {FINAL_ITERS}",
        f"Лучший чекпойнт по val-loss, устройство: {device}",
    ]
    save_text_protocol("protocol_2_model.png", arch_lines, font_size=10)

    tuning_table = [
        "Подбор конфигурации: короткие прогоны по "
        f"{SCREEN_ITERS} шагов, сравнение по val-перплексии",
        "",
        f"  {'имя':<6}{'n_layer':>8}{'n_head':>7}{'n_embd':>7}"
        f"{'параметры':>11}{'val ppl':>9}",
    ]
    for r in tuning_rows:
        mark = "  <- выбрана" if r["name"] == final_cfg["name"] else ""
        tuning_table.append(
            f"  {r['name']:<6}{r['n_layer']:>8}{r['n_head']:>7}{r['n_embd']:>7}"
            f"{r['params']:>11}{r['val_ppl']:>9.2f}{mark}"
        )
    tuning_table += [
        "",
        "Вывод: выбрана конфигурация с минимальной val-перплексией;",
        "она обучена полностью на следующем этапе.",
    ]
    save_text_protocol("protocol_3_tuning.png", tuning_table, font_size=10)

    end_rows = [
        f"шаг {r[0]:>4d}: train loss={r[1]:.4f}, val loss={r[2]:.4f}"
        for r in log_rows[-10:]
    ]
    save_text_protocol(
        "protocol_4_training_end.png",
        [
            f"Полное обучение модели '{final_cfg['name']}' "
            f"({FINAL_ITERS} шагов) — последние оценки:",
            "",
            *end_rows,
            "",
            f"Лучший чекпойнт: шаг {best_iter}, val loss={best_val:.4f}",
            "",
            "Итоговые метрики (детерминированный проход по выборкам):",
            f"  train: loss={train_loss_full:.4f}, perplexity={train_ppl:.2f}",
            f"  val:   loss={val_loss_full:.4f}, perplexity={val_ppl:.2f}",
            "",
            "Перплексия на val против бейзлайнов:",
            f"  unigram = {baselines['unigram']:.2f}",
            f"  bigram  = {baselines['bigram']:.2f}",
            f"  GPT     = {val_ppl:.2f}  (в {gain:.1f} раза лучше биграммы)",
        ],
        font_size=10,
    )

    sample_lines = ["Сгенерированный текст (детерминированно при SEED=42):", ""]
    for s, sample in samples:
        topk_str = s["top_k"] if s["top_k"] else "off"
        sample_lines.append(f"[ temperature={s['temperature']}, top_k={topk_str} ]")
        sample_lines += wrap_sample(sample.strip(), width=64, max_lines=10)
        sample_lines.append("")
    save_text_protocol("protocol_7_samples.png", sample_lines, font_size=9)

    print("Артефакты сохранены:")
    for path in [
        CORPUS_PATH, TRAINING_LOG_PATH, TUNING_LOG_PATH, MODEL_PATH, SAMPLES_PATH,
        LOSS_CURVE_PATH, PERPLEXITY_PATH, ATTENTION_PATH,
        os.path.join(PROTOCOL_DIR, "protocol_1_dataset.png"),
        os.path.join(PROTOCOL_DIR, "protocol_2_model.png"),
        os.path.join(PROTOCOL_DIR, "protocol_3_tuning.png"),
        os.path.join(PROTOCOL_DIR, "protocol_4_training_end.png"),
        os.path.join(PROTOCOL_DIR, "protocol_5_loss_curve.png"),
        os.path.join(PROTOCOL_DIR, "protocol_6_perplexity.png"),
        os.path.join(PROTOCOL_DIR, "protocol_7_samples.png"),
        os.path.join(PROTOCOL_DIR, "protocol_8_attention.png"),
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

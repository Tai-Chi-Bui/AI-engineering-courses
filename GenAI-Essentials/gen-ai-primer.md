# Gen AI Primer — Quick Mental Models

> Short, intuitive explanations of the core mechanisms behind generative AI. For the full deep-dive with code examples, see [fundamentals.md](fundamentals.md).

---

## 1. Tokens and the Context Window

### How text generation works

The decoder takes input ("The quick"), generates a single token, then feeds that output back into itself to predict the next token — building the sequence step by step. This loop is called **autoregressive generation**.

### The 2 costs of longer sequences

- **Memory:** Every token must be stored. More tokens = more RAM, until memory runs out.
- **Compute:** Each new token requires attention over all previous tokens, so longer sequences get progressively slower and more expensive.

**In short:** The longer the text, the more memory and processing power the model needs — which is why LLMs have a **context window limit** (a hard cap on total tokens the model can see at once).

---

## 2. Embeddings

### Vector

Simply a point in space with a direction and magnitude (like an arrow).

### Vector space model

A way to represent text as arrows in multi-dimensional space. Similar meaning = similar direction.

### What are embeddings?

Numbers (vectors) that represent data so ML models can understand relationships between things. For example:
- "King" and "Queen" would have vectors that are close together
- "King" and "Plaza" would be far apart

Think of it like a **map where similar ideas are placed near each other**, so the model can easily find what's related.

### Key properties

- Different embedding algorithms capture different types of relationships — some focus on meaning, others on topic, style, etc.
- Embeddings act as **external memory** for ML models — they encode knowledge the model can look up while performing a task.
- Embeddings can be **shared across models** (multi-model pattern) to help coordinate a task between them.

---

## 3. Positional Encoding

### The problem

Transformers process all words **at the same time** (not one by one). This means they can lose track of word order, which completely changes meaning:

> "Dog bites man" ≠ "Man bites dog"

### The solution

Each word gets a **position tag** added to its embedding before processing. The word's embedding vector gets combined with a position vector, so the model always knows *what* a word is and *where* it sits in the sentence.

### The full processing flow

```
INPUT SENTENCE
      ↓
1. Tokenization         → split sentence into tokens
2. Embedding            → convert tokens into vectors
3. Positional Encoding  → add position info to each vector
      ↓
4. Transformer Blocks (Encoder)
   - Attention
   - Feedforward
      ↓
5. Decoder begins
   - Takes encoded input
   - Generates FIRST output token
      ↓
6. Softmax → picks a token (e.g., "jumps")
      ↓
┌─────────────────────────────────┐
│   AUTOREGRESSIVE LOOP starts   │
│                                 │
│  Output token fed BACK as input │
│         ↓                       │
│  Decoder now sees:              │
│  [original input + "jumps"]     │
│         ↓                       │
│  Generates NEXT token...        │
│         ↓                       │
│  Softmax → picks "over"         │
│         ↓                       │
│  Feed back again...             │
│  [original input + "jumps over"]│
│         ↓                       │
│  ...repeats until END token     │
└─────────────────────────────────┘
```

---

## 4. Attention

### Core idea

Attention assigns a **weight** (importance score) to each token relative to others. Higher weight = more relevant.

### 3 types of attention

**1. Self-attention** — tokens in the *same* sequence looking at each other

> Example: "The animal didn't cross the street because **it** was too tired"
>
> - The model needs to figure out what "it" refers to.
> - Self-attention lets "it" look at all other words and realize → **it = animal**
> - "animal" gets the highest attention weight.

**2. Cross-attention** — tokens from one sequence looking at *another* sequence

> Example — Translation (English → Vietnamese):
> - The decoder (Vietnamese output) looks back at the encoder (English input).
> - "Con meo" knows to pay attention to "cat" from the original.
>
> ```
> Encoder (English):    "The cat sat"
>         ↕ cross-attention
> Decoder (Vietnamese): "Con meo ngoi"
> ```

**3. Multi-head attention** — running multiple attention processes **in parallel**, each focusing on something different

Think of it like multiple people reading the same sentence but each looking for different things:

| Head   | Focuses on                       |
|--------|----------------------------------|
| Head 1 | Grammar relationships            |
| Head 2 | Who is doing what (subject/verb) |
| Head 3 | Tone and sentiment               |
| Head 4 | Long-range dependencies          |

The outputs of all heads are concatenated and combined, giving the model a richer understanding than any single attention pass could provide.

---

## 5. Neural Network Building Blocks

*Understanding these is a prerequisite for understanding fine-tuning.*

### Node (neuron)

A small calculator that:
1. Takes numbers in
2. Does a simple math operation (weighted sum + activation function)
3. Passes a number out

Think of it as a person who **receives opinions from others, weighs them, and forms their own opinion**.

### Connection and weight

Nodes are linked by connections — like wires carrying signals between them.

```
Node A → (connection with weight) → Node B
```

Not all connections are equal. Each has a **weight** — a number that controls how strongly one node influences another.

> Think of weights like trust levels:
> - You trust your doctor's opinion a lot → **high weight**
> - You trust a stranger's opinion less → **low weight**

### Layer

Multiple nodes sitting at the same stage of processing form a layer:

- **Input layer** — sees the raw input
- **Hidden layers** — find patterns
- **Output layer** — produces the result

### Parameters

**Parameters = all the weights** (and biases) in the entire network.

More parameters = more connections = the model can understand more complex things.

---

## 6. Fine-Tuning

### What is it?

During **pre-training**, the model adjusts all its weights by learning from massive amounts of text — billions of web pages, books, etc. After pre-training, the weights are fixed. The model is smart but general.

**Fine-tuning** = taking that pre-trained model and **slightly adjusting specific weights** for a specific job.

### Two dimensions of fine-tuning

#### A. Changing the dataset (what the model learns from)

| Approach | What it does |
|----------|-------------|
| **Instruction fine-tuning** | Train on prompt/response pairs so the model follows instructions |
| **Domain-specific fine-tuning** | Train on specialized data (legal, medical) so the model learns domain language |

#### B. Changing the method (how the weights get updated)

| Method | What it does | Trade-off |
|--------|-------------|-----------|
| **Full fine-tuning** | Update all weights in the entire model | Most powerful, but very expensive |
| **PEFT** (Parameter-Efficient Fine-Tuning) | Freeze most weights, only update a tiny subset | Much cheaper, nearly as effective |
| **LoRA** | Most popular PEFT method — adds small trainable matrices alongside frozen weights instead of changing them directly | Great balance of cost and quality |
| **Last-layer fine-tuning** | Only update the final layer(s) | Cheapest, but least flexible |

#### C. Reducing model size

| Method | What it does |
|--------|-------------|
| **Pruning** (train-time) | Remove unnecessary weights during training |
| **Pruning** (post-training) | Remove unnecessary weights after training to make the model smaller and faster |

# Core LLM Elements — Beginner's Complete Guide

> **Who is this for?** Complete beginners who want to understand how Large Language Models (LLMs) work under the hood — not just how to use them, but *why* they behave the way they do. By the end of this guide, you'll be able to configure LLMs intelligently and debug them when they misbehave.

---

## Table of Contents

1. [Tokens — The Alphabet of LLMs](#1-tokens--the-alphabet-of-llms)
2. [Context Windows — The Model's Working Memory](#2-context-windows--the-models-working-memory)
3. [The Sampling Pipeline — How Text Gets Generated](#3-the-sampling-pipeline--how-text-gets-generated)
4. [Temperature — The Creativity Dial](#4-temperature--the-creativity-dial)
5. [Top-K Sampling — A Blunt Filter](#5-top-k-sampling--a-blunt-filter)
6. [Top-P Sampling — The Smarter Filter](#6-top-p-sampling--the-smarter-filter)
7. [Repetition Penalties — Stopping the Loop](#7-repetition-penalties--stopping-the-loop)
8. [Putting It All Together — Real-World Configurations](#8-putting-it-all-together--real-world-configurations)
9. [Cost Estimation — Thinking in Tokens](#9-cost-estimation--thinking-in-tokens)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Tokens — The Alphabet of LLMs

### What is a token?

When you type a message to an LLM, the model doesn't read your text the way you do — letter by letter or word by word. Instead, it converts your text into a sequence of **tokens**, which are chunks of text of varying sizes.

Think of tokens like Lego bricks. A word might be one brick, or it might be broken into two or three smaller bricks, depending on how common it is.

```
"Hello world!"  →  ["Hello", " world", "!"]  →  [9906, 1917, 0]
```

The model only ever sees those integer IDs — `[9906, 1917, 0]` — never the original text.

### Why subwords, not whole words?

A simple word-based vocabulary would have millions of entries and still couldn't handle:
- Made-up words (e.g., "Anthropic", "ChatGPT")
- Rare technical terms
- Typos and misspellings
- Words in languages the vocabulary didn't anticipate

Subword tokenization solves this by breaking unknown words into known pieces. For example:
```
"tokenization"  →  ["token", "ization"]
"unbelievable"  →  ["un", "believ", "able"]
"ChatGPT"       →  ["Chat", "G", "PT"]
```

Even a word the model has never seen before can be represented using familiar subword pieces.

### Byte-Pair Encoding (BPE) — how vocabulary is built

The most common tokenization algorithm is called **Byte-Pair Encoding (BPE)**. Here's how it works, step by step:

**Step 1:** Start with the smallest possible units — individual characters (or bytes).
```
["h", "e", "l", "l", "o"]
```

**Step 2:** Count every adjacent pair in a large corpus of training text. Find the most frequent pair.
```
Most frequent pair: ("e", "r") appears 50,000 times
```

**Step 3:** Merge that pair into a new single token `"er"`. Add it to the vocabulary.

**Step 4:** Repeat steps 2–3 thousands of times until you reach your desired vocabulary size (typically 32,000–100,000 tokens).

The result: common words like "the", "is", "and" become single tokens. Rare words get split into smaller familiar pieces.

### The ~4 characters per token rule of thumb

On average, for English text, **1 token ≈ 4 characters**. This means:
- 1 word ≈ 1–2 tokens
- 1 sentence ≈ 15–20 tokens
- 1 page of text ≈ 400–500 tokens
- 1,000 words ≈ ~750 tokens

> ⚠️ **This rule only applies to English.** Non-English languages, especially those with non-Latin scripts (Thai, Arabic, Chinese, Korean), are far less efficient. The same sentence in Thai might cost 4× more tokens than in English — because most LLM tokenizers were trained predominantly on English data, so non-English characters get mapped to smaller, less-merged subword units — sometimes individual bytes.

### Special tokens

Beyond regular text tokens, tokenizers include **special tokens** that signal structure to the model:

| Token | Purpose |
|---|---|
| `[BOS]` / `<s>` | Beginning of sequence |
| `[EOS]` / `</s>` | End of sequence — tells model to stop generating |
| `[PAD]` | Padding to make batches equal length |
| `[SEP]` | Separator between segments |
| `<\|system\|>` | Marks start of system prompt (model-specific) |
| `<\|user\|>` | Marks start of user turn |

Understanding special tokens matters when you're working directly with model APIs or fine-tuning.

### Practical exercise

Install tiktoken and try this:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

texts = [
    "Hello world!",
    "Tokenization is fascinating.",
    "สวัสดีชาวโลก",  # "Hello world" in Thai
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
]

for text in texts:
    tokens = enc.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)}")
    print(f"Decoded pieces: {[enc.decode([t]) for t in tokens]}")
    print()
```

Notice how the Thai text uses far more tokens, and how the Python code tokenizes differently from prose.

---

## 2. Context Windows — The Model's Working Memory

### What is the context window?

Every LLM has a maximum number of tokens it can process at once. This is called the **context window** — and it's the model's entire universe for a given request.

```
┌─────────────────────────────────────────────┐
│              Context Window (e.g. 8,192 tokens)              │
│                                                                          │
│  System prompt    │  Conversation history  │  Your message  │  Response  │
│  (300 tokens)     │  (2,000 tokens)        │  (200 tokens)  │  (?)       │
└─────────────────────────────────────────────┘
```

**Critical insight:** The context window is a **shared budget** between input and output. If your input uses 7,000 tokens and the context window is 8,192, only 1,192 tokens remain for the model's response.

> In production, always track both sides of this budget. A verbose system prompt quietly eats into how much the model can reply with.

### What counts as input tokens?

Everything the model sees before it starts generating:
- Your system prompt
- The entire conversation history (all previous messages)
- Tool/function call results
- Retrieved documents (in RAG systems)
- Any files you've attached

### KV Cache — why inference is fast

When you send a 1,000-token prompt and ask for a 200-token reply, the model generates one token at a time. Without optimization, it would need to reprocess all 1,000 prompt tokens on every single generation step — that would be impossibly slow.

The solution is the **KV cache** (Key-Value cache):

1. During prompt processing, the model computes **attention keys and values** for every token
2. These get stored in the KV cache (in GPU memory)
3. On each generation step, only the *new* token needs to be computed — everything else is read from cache

```
Step 1: Process all 1,000 prompt tokens → store KV cache
Step 2: Generate token 1 → compute only token 1, read cache for context
Step 3: Generate token 2 → compute only token 2, read cache for context
...
```

**The catch:** The KV cache grows linearly with context length. At very long contexts (128k+ tokens), the KV cache can consume more GPU memory than the model weights themselves. This is why long-context inference is expensive.

### The lost-in-the-middle problem

Research (Liu et al., 2023) found a surprising weakness: transformers don't pay equal attention to all parts of the context. They tend to pay more attention to:
- Tokens near the **beginning** of the context (primacy bias)
- Tokens near the **end** of the context (recency bias)

Content buried in the **middle** gets underweighted and is more likely to be "forgotten."

```
Attention strength:
HIGH  ▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓  HIGH
      [Beginning]        [Middle]                    [End]
```

**Practical implication for RAG systems:** When stuffing retrieved documents into a prompt, put the most important information either first or last — never in the middle of a long context.

### Bigger context ≠ always better

A common beginner mistake: assuming that a 128k context window means you should stuff everything in. In reality:

| Factor | Impact of larger context |
|---|---|
| Cost | More input tokens = higher API cost |
| Latency | More tokens = slower time-to-first-token |
| Quality | Lost-in-the-middle effect worsens with irrelevant content |

A well-tuned RAG system with **3 highly relevant** documents almost always outperforms one with **30 loosely relevant** ones.

---

## 3. The Sampling Pipeline — How Text Gets Generated

### The big picture

Before we talk about temperature and Top-P, you need to understand *what they operate on*. Here's the complete pipeline:

```
Your prompt
    ↓
[Transformer forward pass]
    ↓
Logits (one raw score per vocabulary token)
    ↓
÷ Temperature
    ↓
Softmax → Probability distribution
    ↓
Top-K filter (optional)
    ↓
Top-P filter (optional)
    ↓
Sample one token
    ↓
Append to context, repeat until [EOS] or max_tokens
```

Understanding this pipeline is the key to understanding every parameter you'll ever tune.

### Step 1: Logits

After processing your prompt, the model outputs a vector of **logits** — one raw score per token in the vocabulary.

```python
# Conceptual example — not real code
logits = model.forward(prompt_tokens)
# logits is a vector of ~50,000 numbers, e.g.:
# [2.1, -0.3, 4.7, 1.2, -2.8, ...]
#  "the"  "a"  "cat"  "sat"  "flew"
```

Logits are **unbounded real numbers** — they can be negative, very large, very small. They are *not* probabilities. A negative logit doesn't mean impossible, just less likely.

### Step 2: Softmax

Softmax converts logits into a proper probability distribution (all values positive, sum to 1):

```
P(token_i) = exp(logit_i) / sum(exp(logit_j) for all j)
```

Simple example with 5 tokens:

| Token | Logit | After softmax |
|---|---|---|
| "cat" | 4.7 | 0.65 |
| "the" | 2.1 | 0.18 |
| "sat" | 1.2 | 0.08 |
| "a" | -0.3 | 0.04 |
| "flew" | -2.8 | 0.01 |
| | **Total** | **1.00** ✓ |

Notice: even "flew" with logit -2.8 gets probability 0.01, not 0. Softmax always assigns some probability to every token.

### Step 3: Sampling strategies

Once you have a probability distribution, you need to pick the next token. There are several strategies:

**Greedy decoding** — always pick the highest probability token. Deterministic and fast, but produces repetitive, boring output for open-ended tasks.

**Beam search** — keep the top K sequences at each step and pick the globally best one. Works well for translation and summarization, but produces generic text for chat.

**Stochastic sampling** — randomly draw from the probability distribution. Used by all modern chat LLMs. This is where temperature, Top-K, and Top-P come in.

---

## 4. Temperature — The Creativity Dial

### The formula

Temperature is applied *before* softmax by dividing all logits by T:

```
adjusted_logit_i = logit_i / T
```

Then softmax is applied to the adjusted logits.

### What temperature does to the distribution

The key insight: dividing logits by T **amplifies or shrinks the gaps** between them.

**Low temperature (T < 1) — sharper, more deterministic:**
```
Original logits:  [4.7,  2.1,  1.2, -0.3, -2.8]
T = 0.3:         [15.7,  7.0,  4.0, -1.0, -9.3]   ← gaps amplified
After softmax:   [0.99,  0.01,  0.00,  0.00,  0.00]  ← one token dominates
```

**High temperature (T > 1) — flatter, more random:**
```
Original logits:  [4.7,  2.1,  1.2, -0.3, -2.8]
T = 2.0:          [2.35, 1.05, 0.60, -0.15, -1.4]   ← gaps shrunk
After softmax:    [0.45, 0.23, 0.15,  0.10,  0.07]  ← more uniform
```

**T = 0** → pure greedy (always picks the top token)
**T = 1** → model's natural distribution, no modification
**T → ∞** → uniform random distribution (all tokens equally likely)

### The temperature intuition

Think of temperature as a **confidence dial**:

- **Low T:** Model is very confident. Says: *"I know exactly what word comes next."*
- **High T:** Model is exploratory. Says: *"Many words could work here, let's try something interesting."*

### Practical temperature guide

| Use case | Recommended T | Reason |
|---|---|---|
| Code generation | 0.0 – 0.2 | Syntax must be correct; creativity is harmful |
| Factual Q&A | 0.1 – 0.3 | Accuracy matters; avoid hallucination |
| Customer support | 0.5 – 0.7 | Consistent but not robotic |
| General chat | 0.7 | Good balance of variety and coherence |
| Creative writing | 0.9 – 1.2 | Variety and surprise are desirable |
| Brainstorming | 1.0 – 1.5 | Maximum diversity of ideas |

> ⚠️ **T > 1.5 in production is almost always a mistake.** At that level, the model starts picking low-probability tokens frequently, leading to incoherent or hallucinated output.

---

## 5. Top-K Sampling — A Blunt Filter

### What it does

After applying temperature, Top-K **throws away all but the K highest-probability tokens** and redistributes all probability mass across those K tokens.

```
Original distribution (after temperature):
Token:        "cat"  "dog"  "bird"  "sat"  "flew"  ... (49,995 more)
Probability:   0.45   0.20   0.15   0.08   0.05    ...  tiny

With K=3: keep only ["cat", "dog", "bird"], renormalize:
Token:        "cat"  "dog"  "bird"
Probability:   0.56   0.25   0.19    (renormalized to sum to 1)
```

### The problem with Top-K

Top-K uses a **fixed** count regardless of the shape of the distribution. This causes two failure modes:

**Scenario A: Distribution is very concentrated (model is confident)**
```
"cat":  0.92
"dog":  0.06
"bird": 0.01
... 49,997 tokens with ~0.00 probability
```
With K=50, you're still including 47 low-quality tokens in your sample pool when only "cat" and maybe "dog" are sensible choices.

**Scenario B: Distribution is very flat (model is uncertain)**
```
500 tokens each with probability ~0.002
```
With K=50, you're cutting out 450 perfectly reasonable options arbitrarily.

Top-K is blind to context — it doesn't know whether 50 tokens is too many or too few for the current situation. This rigidity is its fundamental weakness, and why it's often disabled in favor of Top-P.

---

## 6. Top-P Sampling — The Smarter Filter

### What it does

Top-P (also called **nucleus sampling**) fixes Top-K's rigidity with a simple insight: instead of a fixed *count*, use a fixed *probability mass*.

**Algorithm:**
1. Sort tokens by probability, highest first
2. Accumulate tokens until their cumulative probability reaches P
3. Sample from that set (the "nucleus")

```
Example with P=0.9:

Token      Probability   Cumulative
"cat"         0.45          0.45
"dog"         0.20          0.65
"bird"        0.15          0.80
"sat"         0.08          0.88
"the"         0.05          0.93  ← crossed 0.90 here
----- nucleus contains: cat, dog, bird, sat, the -----
```

### Why Top-P adapts where Top-K can't

| Scenario | Top-K (K=10) | Top-P (P=0.9) |
|---|---|---|
| Model very confident (3 tokens have 95% mass) | Still includes 7 low-quality tokens | Stops at 3 tokens — tight nucleus |
| Model uncertain (500 tokens share mass evenly) | Cuts off 490 valid options | Includes hundreds of tokens — wide nucleus |

The nucleus **contracts** when the model is sure and **expands** when the model is uncertain. This is why Top-P with P=0.9 is the default in most production systems.

### Top-K vs Top-P: which to use

- **Use Top-P** as your primary vocabulary filter (P=0.9 or P=0.95 is a sensible default)
- **Disable Top-K** (set K=0) when using Top-P — combining them can overly restrict the vocabulary
- The only time Top-K is useful is when you specifically want a hard cap on vocabulary diversity regardless of confidence

### How temperature and Top-P interact

Remember the pipeline order: **temperature first, then Top-P**.

- High temperature → flatter distribution → more tokens needed to reach P → wider nucleus
- Low temperature → sharper distribution → fewer tokens needed to reach P → tighter nucleus

This means T and Top-P are not independent knobs. Raising T effectively expands the nucleus even if P stays the same.

---

## 7. Repetition Penalties — Stopping the Loop

### Why LLMs repeat themselves

LLMs have a natural tendency to repeat tokens that have already appeared in the context. This isn't a bug — it's learned behavior. Human text *does* repeat words and phrases naturally. The model learned that "once a word appears, it's likely to appear again" from its training data.

The problem: during long generation, this tendency **compounds**. Once a token appears, its probability gets a slight boost, making it more likely to appear again, boosting it again, until the model is stuck in a loop like:

> *"The key takeaway is that the key takeaway is that the key takeaway is that..."*

Repetition penalties are the fix.

### How penalties work

All repetition penalties work by **reducing the logit of tokens that have already appeared** before sampling.

There are three main flavors:

---

#### OpenAI: Presence Penalty

**Binary penalty** — applied once, regardless of how many times a token appeared.

```
If token appeared in context: logit -= presence_penalty
If token hasn't appeared:     logit unchanged
```

Range: -2.0 to 2.0 (negative values *encourage* repetition)
Typical value: 0.1 – 0.6

**Best for:** Encouraging topic diversity — discouraging the model from circling back to the same concepts.

---

#### OpenAI: Frequency Penalty

**Count-scaled penalty** — grows with each additional occurrence.

```
Penalty = frequency_penalty × (number of times token appeared)
logit -= penalty
```

Range: -2.0 to 2.0
Typical value: 0.1 – 0.5

**Best for:** Suppressing true loops where the same word keeps getting picked over and over.

**Key difference from presence penalty:**

| Token | Appeared | Presence penalty (0.5) | Frequency penalty (0.5) |
|---|---|---|---|
| "the" | 10 times | -0.5 (once) | -5.0 (×10) |
| "cat" | 1 time | -0.5 (once) | -0.5 (×1) |

Frequency penalty becomes very aggressive for highly repeated tokens. Set it above ~0.8 and common words like "the", "is", "and" get penalized so hard the output becomes bizarre.

---

#### Hugging Face: repetition_penalty

Different implementation — **divides** the logit rather than subtracting from it.

```
If token appeared: logit = logit / repetition_penalty
```

Default: 1.0 (divide by 1 = no change)
Typical useful range: 1.1 – 1.3

Example:
```
Token logit: 2.6
repetition_penalty = 1.3
New logit: 2.6 / 1.3 ≈ 2.0  (reduced, less likely to be sampled)
```

> ⚠️ **Important:** This applies to all previously seen tokens, *including the prompt*. If your prompt contains many repeated keywords (e.g., a long document about "machine learning"), those keywords get penalized in the output too. Keep `repetition_penalty` below 1.3 to avoid harming coherence.

### When to use repetition penalties

| Situation | Recommendation |
|---|---|
| Short responses (< 200 tokens) | Usually not needed |
| Long-form content generation | frequency_penalty 0.1–0.3 |
| Storytelling / creative writing | frequency_penalty 0.3–0.5 or repetition_penalty 1.1–1.2 |
| Already getting loops at T=0 | Raise temperature first, then add penalty |
| Seeing bizarre avoidance of common words | Your penalty is too high — reduce it |

---

## 8. Putting It All Together — Real-World Configurations

### The full pipeline (revisited)

```
Logits
  ↓  ÷ Temperature           (reshape distribution)
  ↓  Top-K filter            (optional hard truncation)
  ↓  Top-P filter            (optional nucleus cutoff)
  ↓  Repetition penalty      (downweight already-seen tokens)
  ↓  Sample
  ↓  Append, repeat
```

### Configuration recipes for common tasks

#### Code autocomplete
```python
temperature = 0.1
top_p = 0.95
top_k = 0          # disabled
repetition_penalty = 1.0   # disabled
```
*Rationale: Code has correct answers. Be deterministic. No penalty needed — code naturally doesn't repeat.*

#### Customer support chatbot
```python
temperature = 0.7
top_p = 0.9
top_k = 0
repetition_penalty = 1.1   # mild, prevents phrase loops in long conversations
```
*Rationale: Natural variety without going off-topic. Mild penalty for long conversations.*

#### Creative story generation
```python
temperature = 1.0
top_p = 0.95
top_k = 0
frequency_penalty = 0.4    # prevent repetitive descriptions
```
*Rationale: Natural distribution, wide nucleus, penalize repetitive language.*

#### Factual summarization
```python
temperature = 0.2
top_p = 0.9
top_k = 0
repetition_penalty = 1.0
```
*Rationale: Stay close to source material. Low temperature = faithful, not creative.*

#### Brainstorming / ideation
```python
temperature = 1.2
top_p = 0.99
top_k = 0
repetition_penalty = 1.15
```
*Rationale: High temperature for diversity. Penalty prevents cycling through same ideas.*

### Debugging guide: reading symptoms

| Symptom | Likely cause | Fix |
|---|---|---|
| Always same output, robotic | Temperature too low | Raise T to 0.7+ |
| Incoherent, random gibberish | Temperature too high | Lower T below 1.2 |
| Repeating phrases in loops | No repetition penalty | Add frequency_penalty 0.2–0.4 |
| Avoids common words unnaturally | Penalty too high | Lower penalty |
| Cuts off mid-sentence | max_tokens too low | Increase max_tokens |
| Missing key info from prompt | Lost-in-the-middle | Move key info to beginning or end |
| Hallucinating facts | Temperature too high for factual task | Lower T, use RAG |

---

## 9. Cost Estimation — Thinking in Tokens

### How API pricing works

Most LLM APIs charge separately for:
- **Input tokens** — everything in your prompt
- **Output tokens** — everything the model generates

Output tokens typically cost **5–10× more** than input tokens. This surprises many beginners who focus only on prompt length.

### Cost calculation formula

```
Daily cost = (input_tokens_per_call × calls_per_day × input_price_per_M)
           + (output_tokens_per_call × calls_per_day × output_price_per_M)
```

### Worked example

**Setup:**
- System prompt: 2,000 tokens
- Average user message: 200 tokens
- Average response: 500 tokens
- Daily calls: 10,000
- Input price: $3 per million tokens
- Output price: $15 per million tokens

**Calculation:**
```
Input:  (2,000 + 200) × 10,000 = 22,000,000 tokens → 22M × $3/1M  = $66/day
Output: 500 × 10,000           =  5,000,000 tokens →  5M × $15/1M = $75/day
                                                        Total: $141/day = ~$4,230/month
```

**Key insight:** Despite being far fewer tokens, output costs *more* than input in this example ($75 vs $66). Output tokens are expensive — always set sensible `max_tokens` limits.

### Cost optimization strategies

1. **Shorten system prompts** — every token in your system prompt multiplies across all calls
2. **Set `max_tokens` limits** — prevent runaway long responses
3. **Cache common prompts** — many providers offer prompt caching for repeated system prompts
4. **Choose the right model** — smaller/cheaper models for simple tasks
5. **Batch requests** — many providers offer batch APIs at 50% discount for non-real-time workloads

### Important: `max_tokens` is a cap, not a reservation

A common misconception: that setting `max_tokens=1000` means you're billed for 1000 output tokens. **You're not.**

- `max_tokens` is the maximum the model is *allowed* to generate
- You're billed for actual tokens generated
- If the model finishes naturally at 300 tokens, you pay for 300

Set `max_tokens` generously and let the model's natural stop token do the work. Setting it too low risks truncating responses mid-sentence.

---

## 10. Quick Reference Cheat Sheet

### Key concepts at a glance

| Concept | One-line definition |
|---|---|
| Token | The atomic unit an LLM processes; roughly 4 chars of English text |
| Tokenizer | Algorithm that converts text ↔ token IDs |
| BPE | Algorithm that builds vocabulary by repeatedly merging frequent character pairs |
| Context window | Total token budget (input + output) the model can see at once |
| KV cache | GPU memory cache that stores attention keys/values to avoid recomputation |
| Lost-in-the-middle | Model's tendency to underweight information in the middle of long contexts |
| Logits | Raw unbounded scores output by the model before softmax |
| Softmax | Converts logits into a probability distribution summing to 1 |
| Temperature | Divides logits before softmax; lower = sharper, higher = flatter |
| Top-K | Hard-keeps only the K highest-probability tokens |
| Top-P | Keeps tokens until cumulative probability reaches P (adaptive vocabulary) |
| Presence penalty | Fixed one-time penalty for tokens that appeared in context |
| Frequency penalty | Cumulative penalty that grows with each repeated occurrence |
| repetition_penalty | Hugging Face param; divides logit of repeated tokens |

### The sampling pipeline in one line

```
logits → ÷T → softmax → Top-K → Top-P → repetition penalty → sample
```

### Temperature reference

| T value | Effect | Use for |
|---|---|---|
| 0.0 | Greedy (deterministic) | Debug, exact tasks |
| 0.1 – 0.3 | Very focused | Code, factual Q&A |
| 0.5 – 0.7 | Balanced | Chat, support bots |
| 0.9 – 1.2 | Creative | Stories, copywriting |
| > 1.5 | Chaotic | Almost never useful |

### Top-P reference

| P value | Effect |
|---|---|
| 0.5 | Very conservative, small nucleus |
| 0.9 | Standard recommended default |
| 0.95 | Slightly more diverse |
| 1.0 | No cutoff — relies entirely on temperature |

### Repetition penalty reference

| Parameter | No penalty | Mild | Strong | Too strong |
|---|---|---|---|---|
| OpenAI frequency_penalty | 0.0 | 0.2 | 0.6 | > 1.0 |
| OpenAI presence_penalty | 0.0 | 0.3 | 0.8 | > 1.5 |
| HF repetition_penalty | 1.0 | 1.1 | 1.3 | > 1.5 |

---

## What to Study Next

Now that you understand Core LLM Elements, you're ready for:

1. **Prompt Engineering** — using your knowledge of tokens and sampling to write better prompts
2. **RAG (Retrieval-Augmented Generation)** — managing context windows intelligently at scale
3. **Fine-tuning** — adjusting model weights rather than just sampling parameters
4. **LLM Evaluation** — measuring output quality systematically rather than tuning by feel
5. **LLMOps** — monitoring token costs, latency, and quality in production

---

*Guide written based on the roadmap.sh AI Engineer curriculum. All parameter recommendations are guidelines — always validate empirically for your specific use case.*

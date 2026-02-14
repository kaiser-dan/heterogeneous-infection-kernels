# Twitter Contagion Data & Code Resource Guide

## 1. Dataset Overview
The dataset contains data from a controlled experiment on Twitter designed to test complex contagion hypotheses. It consists of bot activity, user retweets, and the underlying social graph.

### `all_bot_tweets_` (JSON)
- **Content**: A flat list of Tweet objects representing the *original* tweets posted by the bots.
- **Structure**: Standard Twitter v1.1 Tweet Object.
  - `id`: Tweet ID (Int64).
  - `user`: User object (Bot).
  - `entities.hashtags`: Contains the intervention hashtag (e.g., `#GoldenGateAliens`).
  - `created_at`: Timestamp string.

### `retweets_` (JSON)
- **Content**: A list of **Cascades**.
- **Structure**: `[[Tweet, Tweet, ...], [Tweet, Tweet, ...]]`
  - Each inner list corresponds to one original bot tweet.
  - The list contains **Retweet Objects**.
  - **Important**: The first element of the list usually contains the reference to the original tweet via `retweeted_status`.
  - These are the "infection events".

### `edges` (CSV)
- **Content**: The social graph (who follows whom).
- **Schema**:
  - `account_id`: The user who is following (Source).
  - `bot_id`: The user/bot being followed (Target).
  - `relation`: "follower" (User -> Bot), "friend" (Bot -> User).
  - `first_seen`, `last_seen`: Timestamps indicating when this edge was observed.
  - **Note**: The column names `account_id` and `bot_id` are slightly misleading; they represent the two endpoints of an edge. The `relation` field dictates direction.

### `targets.db` (SQLite)
- **Content**: Metadata about the users.
- **Schema**:
  - `targets` table: `id_str`, `handle`, `location`...
  - Maps integer IDs to human-readable handles.

---

## 2. Monsted Code Analysis (`twitter_contagion_code/`)

The original code (Monsted et al.) focuses on a **Likelihood-based inference** rather than constructing time-varying $N$ and $M$ matrices directly.

### Key Logic: `tools.py` & `compute_bic_.py`
The Monsted approach computes the probability of a user retweeting given their **Cumulative Exposure**.

1.  **Exposure Collection**:
    - For a specific intervention, it iterates through all participants.
    - It finds all **Exposers**: Unique neighbors who retweeted *before* the target user.
    - It counts the **Multiplicity** of exposure: How many times did neighbor $X$ tweet about this?

2.  **Probabilistic Exposure ($q$)**:
    - The code does **not** assume `Neighbor Infected = You Exposed`.
    - It assumes a tweet in your timeline has probability $q$ of being seen.
    - If neighbor $X$ tweets $k$ times, probability you saw *at least one* is $1 - (1-q)^k$.
    - Distribution: The code calculates the probability distribution $P(K_{eff})$ that you effectively saw $K_{eff}$ *unique* infected neighbors, given your raw exposures.

3.  **Inference**:
    - It maximizes likelihood $L = \sum \log P(\text{Action} | \text{Exposures})$.
    - It compares Simple Contagion (prob depends on ANY exposure) vs Complex Contagion (prob depends on K exposures).

### `src/data_processing/twitter_contagion.py` (Current Attempt)
The current script attempts to bridge the gap but makes a simplification:
- **Deterministic Exposure**: It assumes if a neighbor retweets, you are exposed immediately.
- **Time-Resolved**: It correctly breaks the cascade into time bins (e.g., hours).
- **N/M Construction**: It counts "Successes" ($M$) and "Failures" ($N$) at each time step.
- **Discrepancy**: It ignores the $q$ parameter and the "multiple tweets from same neighbor" nuance, treating the neighbor's state as the sole factor.

---

## 3. Structural Comparison for Proposed Pipeline

| Feature | Monsted (Original) | Target (N/M Matrices) |
| :--- | :--- | :--- |
| **Time** | Collapsed (Pre-infection vs Post-infection) | **Discrete Step** (e.g., Hour $t$) |
| **State** | Binary (Exposed/Not) | **Dose-Response** (Active Neighbors $k$) |
| **Exposure** | Probabilistic ($q$) | **Needs definition** (Likely Deterministic or Thresholded) |
| **Output** | Likelihood Score (BIC) | **Matrices** ($N_{nodes \times k}$, $M_{nodes \times k}$) |

### Recommendation for Pipeline
To generate $N$ and $M$ matrices consistent with the new research goal ("Heterogeneous Infection Kernels"):
1.  **Adopt the Time-Resolved logic** from `twitter_contagion.py`.
2.  **Refine Exposure**: To respect the "Probabilistic Exposure" concern without adding latent variables:
    -   **Option A**: Stick to $k_{active}$ (Active Neighbors). This assumes $q \approx 1$ or is absorbed into the kernel learning.
    -   **Recency Filter**: Only count a neighbor as "Exposing" if they retweeted within the last $X$ hours. This mimics the "feed scrolling" mechanic (posts disappear).

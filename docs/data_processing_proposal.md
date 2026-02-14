# Proposal: Twitter Contagion Data Processing Pipeline

## 1. Goal
Extract discrete-time "Success" ($M$) and "Failure" ($N$) matrices from the raw Twitter contagion dataset and store them in the existing SQLite database. The pipeline will also support aggregating these matrices across multiple cascades to create a unified dataset for kernel inference.

## 2. Core Logic: Exposure & Recency
We will adopt a **Time-Resolved Deterministic Model**.
-   **Standard**: `Exposure = Active Neighbor Count` (cumulative).
-   **Recency Filter (Optional)**: If enabled, a neighbor's "active status" expires after $W$ hours. This mimics the transient visibility of tweets in a feed.

## 3. Implementation Plan

### Phase 1: Extend Existing Logic (`src/data_processing/`)
We will leverage the existing `ORM.py` and `database_utils.py` which already handle blob storage for matrices.

**Modifications to `twitter_contagion.py`**:
-   Update `compute_exposure_counts` to accept a `recency_window` parameter.
-   Implement the sliding window logic for neighbor counts.

### Phase 2: Ingestion Script (`src/data_processing/process_twitter_data.py`)
A new script to drive the ETL process:
1.  **Parse Raw Data**: Load `all_bot_tweets_`, `retweets_`, `edges`.
2.  **Process Cascades**: Iterate through each cascade:
    -   Compute $N$ and $M$ matrices.
    -   Compute the cascade-specific adjacency (subgraph).
    -   Use `database_utils.insert_twitter_run` to store these as compressed binary blobs in the DB.

### Phase 3: Aggregation
To analyze the dataset as a whole:
1.  **Load**: Retrieve N/M blobs for all target runs (e.g., all `GoldenGateAliens` cascades).
2.  **Align & Sum**: Use `NodeMapping` to align matrix rows (Users) and sum the counts:
    -   $M_{total} = \sum M_{cascade}$
    -   $N_{total} = \sum N_{cascade}$
3.  **Store**: Save the aggregated result as a new "Summary Run" in the database.

## 4. Verification
-   **Conservation of Mass**: Total of $M$ matrix must equal total retweets.
-   **DB Integrity**: Ensure blobs can be round-tripped (compressed/decompressed) without data loss.

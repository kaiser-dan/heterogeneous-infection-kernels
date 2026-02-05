from pathlib import Path

# Paths
RAW_OO = config.get("oo_raw", "data/raw/operation_outbreak/oo-histories-all.csv")
RAW_TWITTER = config.get("twitter_raw", "data/raw/twitter")

PROCESSED_BASE = Path("data/processed")

SIM_IDS = [165, 57, 118, 70, 76, 131, 66, 61, 72, 67]  # Top 10 by size (verified with infection data)

rule process_oo_sim:
    """
    Process a single Operation Outbreak simulation.
    Produces exposure history, contact edges, infection nodes, and M/N matrices.
    """
    input:
        histories = RAW_OO,
        script = "src/data_processing/operation_outbreak.py"
    output:
        exposures = PROCESSED_BASE / "oo" / "sim_{sim_id}_exposures.parquet",
        edges = PROCESSED_BASE / "oo" / "sim_{sim_id}_edges.parquet",
        nodes = PROCESSED_BASE / "oo" / "sim_{sim_id}_nodes.parquet",
        m_matrix = PROCESSED_BASE / "oo" / "sim_{sim_id}_M.parquet",
        n_matrix = PROCESSED_BASE / "oo" / "sim_{sim_id}_N.parquet"
    shell:
        "uv run python {input.script} --histories {input.histories} --sim-id {wildcards.sim_id} --output {output.exposures}"



rule process_twitter_dataset:
    """
    Unified rule to process Twitter data using the robust, verified pipeline.
    Produces M/N matrices (sparse/dense) and A/B files directly.
    """
    input:
        data_dir = RAW_TWITTER,
        script = "src/data_processing/process_twitter.py"
    output:
        directory(PROCESSED_BASE / "twitter")
    shell:
        "uv run python {input.script} --data-dir {input.data_dir} --output-dir {output} --no-bot-retweets"

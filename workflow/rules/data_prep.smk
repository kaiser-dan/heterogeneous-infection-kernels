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
    Supports optional backbone filtering.
    """
    input:
        histories = RAW_OO,
        script = "src/data_processing/operation_outbreak.py"
    output:
        exposures = PROCESSED_BASE / "oo" / "sim_{sim_id}_backbone_{method}" / "exposures.parquet",
        edges = PROCESSED_BASE / "oo" / "sim_{sim_id}_backbone_{method}" / "edges.parquet",
        nodes = PROCESSED_BASE / "oo" / "sim_{sim_id}_backbone_{method}" / "nodes.parquet",
        m_matrix = PROCESSED_BASE / "oo" / "sim_{sim_id}_backbone_{method}" / "M.parquet",
        n_matrix = PROCESSED_BASE / "oo" / "sim_{sim_id}_backbone_{method}" / "N.parquet"
    params:
        kmax = lambda wildcards: config.get('oo', {}).get('kmax', 20),
        threshold = lambda wildcards: config.get('oo', {}).get('backbone_threshold', 0.3),
        freq = lambda wildcards: config.get('oo', {}).get('time_bin_freq', '30min'),
        mn_mode = lambda wildcards: config.get('oo', {}).get('mn_mode', 'survival')
    shell:
        "uv run python {input.script} "
        "--histories {input.histories} "
        "--sim-id {wildcards.sim_id} "
        "--output data/processed/oo/sim_{wildcards.sim_id}_backbone_{wildcards.method} "
        "--freq {params.freq} "
        "--kmax {params.kmax} "
        "--mn-mode {params.mn_mode} "
        "--backbone-method {wildcards.method} "
        "--backbone-threshold {params.threshold}"



rule process_twitter_dataset:
    """
    Unified rule to process Twitter data using the robust, verified pipeline.
    Produces M/N matrices (sparse/dense) and A/B files directly.
    """
    input:
        data_dir = RAW_TWITTER,
        script = "src/data_processing/process_twitter.py"
    output:
        directory(Path(TWITTER_PROCESSED_DIR))
    params:
        q_sc = lambda wildcards: config.get('twitter', {}).get(
            'q_sc', data_prep_data.get('twitter', {}).get('q_sc', 0.192)
        ),
        q_cc = lambda wildcards: config.get('twitter', {}).get(
            'q_cc', data_prep_data.get('twitter', {}).get('q_cc', 0.180)
        ),
        mn_mode = lambda wildcards: config.get('twitter', {}).get(
            'mn_mode', data_prep_data.get('twitter', {}).get('mn_mode', 'incidence')
        ),
        full_only_flag = lambda wildcards: (
            "--full-only" if bool(
                config.get('twitter', {}).get(
                    'full_only', data_prep_data.get('twitter', {}).get('full_only', False)
                )
            ) else ""
        )
    shell:
        "uv run python -m src.data_processing.process_twitter "
        "--data-dir {input.data_dir} "
        "--output-dir {output} "
        "--q-sc {params.q_sc} "
        "--q-cc {params.q_cc} "
        "--mn-mode {params.mn_mode} "
        "{params.full_only_flag} "
        "--no-bot-retweets"

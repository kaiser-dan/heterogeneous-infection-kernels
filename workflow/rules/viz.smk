INTERVENTIONS = [
    'getyourflushot', 'highfiveastranger', 'somethinggood',
    'HowManyPushups', 'turkeyface', 'SFThanks',
    'blackfridaystories', 'BanksySF'
]

OO_SIMS = [118, 131, 165]
OO_BACKBONES = ["metric_distance_backbone", "disparity", "lans"]

rule prepare_twitter_dashboard:
    input:
        # Individual interventions
        expand("data/processed/twitter/{intervention}/M_k_q1.0.parquet", intervention=INTERVENTIONS),
        expand("data/processed/twitter/{intervention}/N_k_q1.0.parquet", intervention=INTERVENTIONS),
        # Aggregated full
        "data/processed/twitter/full/M_k_q1.0.parquet",
        "data/processed/twitter/full/N_k_q1.0.parquet"
    output:
        "data/dashboard/dashboard_twitter_data.json"
    shell:
        "python3 src/viz/twitter_prep.py"

rule prepare_oo_dashboard:
    input:
        expand("data/processed/oo/sim_{sim}_backbone_{method}_exposures.parquet", 
               sim=OO_SIMS, method=OO_BACKBONES)
    output:
        "data/dashboard/dashboard_oo_data.json"
    shell:
        "uv run python src/viz/oo_prep.py"

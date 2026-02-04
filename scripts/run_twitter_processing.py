from pathlib import Path
import sys
import argparse

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.process_twitter import build_exposure_matrices

def main():
    parser = argparse.ArgumentParser(description="Process Twitter data into M/N matrices.")
    parser.add_argument("--q", type=float, default=1.0, help="Sensing probability (0.0 to 1.0)")
    parser.add_argument("--count_retweets", action="store_true", default=True, help="Count retweets as exposures (default: True)")
    
    args = parser.parse_args()
    
    data_dir = REPO_ROOT / "data" / "twitter_contagion_data"
    output_dir = REPO_ROOT / "data" / "processed" / "twitter"
    
    print(f"Processing Twitter data with q={args.q}, count_retweets={args.count_retweets}")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    
    build_exposure_matrices(
        data_dir=data_dir,
        output_base_dir=output_dir,
        q=args.q,
        count_retweets_as_exposures=args.count_retweets
    )
    
    print("Processing complete.")

if __name__ == "__main__":
    main()

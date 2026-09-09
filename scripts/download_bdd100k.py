#!/usr/bin/env python3
"""
BDD100K Dataset Downloader from Hugging Face (dgural/bdd100k).
Checks if local dataset already exists, avoids redundant downloads,
extracts metadata, and downloads representative condition images.
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download BDD100K dataset from Hugging Face")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to download")
    parser.add_argument("--download-all", action="store_true", help="Download all images in dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_bdd100k(config: dict, max_samples: int = None, download_all: bool = False, force: bool = False):
    dataset_cfg = config.get("dataset", {})
    repo_id = dataset_cfg.get("name", "dgural/bdd100k")
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    images_dir = root_dir / "images"
    annotations_dir = root_dir / "annotations"
    metadata_dir = root_dir / "metadata"
    splits_dir = root_dir / "splits"

    for d in [images_dir, annotations_dir, metadata_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    local_samples_json = annotations_dir / "samples.json"

    logger.info(f"Connecting to Hugging Face Hub for repo: {repo_id}")
    if not local_samples_json.exists() or force:
        logger.info("Downloading samples.json from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename="samples.json",
            repo_type="dataset"
        )
        shutil.copy2(downloaded_path, local_samples_json)
        logger.info(f"Saved samples.json to {local_samples_json}")
    else:
        logger.info(f"Using existing local samples.json at {local_samples_json}")

    with open(local_samples_json, "r") as f:
        raw_data = json.load(f)

    samples = raw_data.get("samples", raw_data)
    total_samples = len(samples)
    logger.info(f"Total samples recorded in metadata: {total_samples}")

    # Condition distribution mapping to select balanced subset
    # 0=clear, 1=rain, 2=fog, 3=night, 4=dawn_dusk
    condition_buckets = {"clear": [], "rain": [], "fog": [], "night": [], "dawn_dusk": [], "other": []}
    
    for s in samples:
        weather = (s.get("weather") or {}).get("label", "").lower()
        timeofday = (s.get("timeofday") or {}).get("label", "").lower()
        
        if weather == "foggy":
            condition_buckets["fog"].append(s)
        elif weather == "rainy":
            condition_buckets["rain"].append(s)
        elif timeofday == "night":
            condition_buckets["night"].append(s)
        elif timeofday in ["dawn/dusk", "dusk", "dawn"]:
            condition_buckets["dawn_dusk"].append(s)
        elif timeofday == "daytime" and weather in ["clear", "partly cloudy", "overcast"]:
            condition_buckets["clear"].append(s)
        else:
            condition_buckets["other"].append(s)

    logger.info("Condition distribution in full metadata:")
    for cond, bucket in condition_buckets.items():
        logger.info(f"  {cond}: {len(bucket)} samples")

    # Determine subset to download
    if download_all:
        selected_samples = samples
    else:
        limit = max_samples or dataset_cfg.get("max_samples", 1200)
        # Allocate proportionally, always including 100% of fog
        num_fog = len(condition_buckets["fog"])
        remaining_budget = max(0, limit - num_fog)
        per_class = remaining_budget // 4

        selected_samples = []
        selected_samples.extend(condition_buckets["fog"])  # Keep all fog
        selected_samples.extend(condition_buckets["rain"][:per_class])
        selected_samples.extend(condition_buckets["night"][:per_class])
        selected_samples.extend(condition_buckets["dawn_dusk"][:per_class])
        selected_samples.extend(condition_buckets["clear"][:per_class])

    logger.info(f"Selected {len(selected_samples)} samples for benchmark dataset.")

    # Save selected subset manifest
    selected_manifest_path = metadata_dir / "selected_samples.json"
    with open(selected_manifest_path, "w") as f:
        json.dump({"samples": selected_samples}, f, indent=2)

    # Download images concurrently
    num_workers = dataset_cfg.get("download_workers", 16)
    missing_files = []
    for s in selected_samples:
        rel_path = s.get("filepath", "")
        if not rel_path:
            continue
        fname = os.path.basename(rel_path)
        dest = images_dir / fname
        if force or not dest.exists() or dest.stat().st_size == 0:
            missing_files.append((rel_path, dest))

    logger.info(f"Existing images in {images_dir}: {len(selected_samples) - len(missing_files)} / {len(selected_samples)}")
    if missing_files:
        logger.info(f"Downloading {len(missing_files)} missing images using {num_workers} parallel workers...")

        def _download_worker(item):
            hf_rel_path, local_path = item
            try:
                # Direct CDN URL avoids API token rate limits
                cdn_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{hf_rel_path}"
                resp = requests.get(cdn_url, headers={"User-Agent": "Mozilla/5.0 (Research-Benchmark)"}, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(local_path, "wb") as f_out:
                        f_out.write(resp.content)
                    return True
                else:
                    logger.warning(f"Unexpected status {resp.status_code} for {hf_rel_path}")
                    return False
            except Exception as e:
                logger.warning(f"Failed downloading {hf_rel_path}: {e}")
                return False

        import requests
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_worker, item) for item in missing_files]
            success_count = 0
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                if fut.result():
                    success_count += 1
        logger.info(f"Downloaded {success_count}/{len(missing_files)} images successfully.")
    else:
        logger.info("All selected images already present locally.")

    # Calculate dataset size on disk
    total_bytes = sum(f.stat().st_size for f in root_dir.glob("**/*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    # Generate data/bdd100k/dataset_info.txt
    dataset_info_file = root_dir / "dataset_info.txt"
    with open(dataset_info_file, "w") as f:
        f.write("============================================================\n")
        f.write("BDD100K DATASET INFORMATION\n")
        f.write("============================================================\n")
        f.write(f"Source Repository: {repo_id}\n")
        f.write(f"Local Storage Path: {root_dir.resolve()}\n")
        f.write(f"Total Available Samples in Metadata: {total_samples}\n")
        f.write(f"Downloaded Research Benchmark Samples: {len(selected_samples)}\n")
        f.write(f"Total Disk Usage: {total_mb:.2f} MB\n")
        f.write("\nCondition Breakdown (Selected Subset):\n")
        cond_counts = {}
        for s in selected_samples:
            w = (s.get("weather") or {}).get("label", "").lower()
            t = (s.get("timeofday") or {}).get("label", "").lower()
            c = "other"
            if w == "foggy": c = "fog"
            elif w == "rainy": c = "rain"
            elif t == "night": c = "night"
            elif t in ["dawn/dusk", "dusk", "dawn"]: c = "dawn_dusk"
            elif t == "daytime" and w in ["clear", "partly cloudy", "overcast"]: c = "clear"
            cond_counts[c] = cond_counts.get(c, 0) + 1
        for c, count in sorted(cond_counts.items()):
            f.write(f"  {c:12s}: {count}\n")
        f.write("============================================================\n")

    logger.info(f"Saved dataset info to {dataset_info_file}")
    with open(dataset_info_file, "r") as f:
        print(f.read())


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    download_bdd100k(cfg, max_samples=args.max_samples, download_all=args.download_all, force=args.force)

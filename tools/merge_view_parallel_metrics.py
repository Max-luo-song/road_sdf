import argparse
import json
import os
from typing import Dict, List, Optional


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def latest_json(metrics_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(metrics_dir):
        return None
    candidates = [
        os.path.join(metrics_dir, name)
        for name in os.listdir(metrics_dir)
        if name.startswith(prefix) and name.endswith(".json")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_json(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)


def merge_metric_dicts(metric_dicts: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
    merged = {}
    keys = sorted(set().union(*[d.keys() for d in metric_dicts]))
    for key in keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for metrics, weight in zip(metric_dicts, weights):
            if key not in metrics:
                continue
            value = metrics[key]
            if not isinstance(value, (int, float)) or value < 0:
                continue
            weighted_sum += float(value) * weight
            total_weight += weight
        merged[key] = weighted_sum / total_weight if total_weight > 0 else -1
    return merged


def merge_one_prefix(
    group_metric_dirs: List[str],
    output_dir: str,
    prefix: str,
    weights: List[float],
) -> Optional[str]:
    paths = [latest_json(metrics_dir, prefix) for metrics_dir in group_metric_dirs]
    if any(path is None for path in paths):
        print(f"Skipping {prefix}: missing metrics file in one or more groups: {paths}")
        return None

    metric_dicts = [load_json(path) for path in paths]
    merged = merge_metric_dicts(metric_dicts, weights)
    merged["_view_parallel_merge"] = {
        "source_files": paths,
        "weights": weights,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_view_parallel_merged.json")
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged metrics: {output_path}")
    return output_path


def main(args):
    group0_views = parse_int_list(args.group0_views)
    group1_views = parse_int_list(args.group1_views)
    weights = [len(group0_views), len(group1_views)]
    total = sum(weights)
    normalized_weights = [weight / total for weight in weights]

    group_metric_dirs = [args.group0_metrics_dir, args.group1_metrics_dir]
    prefixes = [prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip()]
    for prefix in prefixes:
        merge_one_prefix(
            group_metric_dirs=group_metric_dirs,
            output_dir=args.output_dir,
            prefix=prefix,
            weights=normalized_weights,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge view-parallel evaluation metrics")
    parser.add_argument("--group0_metrics_dir", required=True, type=str)
    parser.add_argument("--group1_metrics_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--group0_views", default="0,2,5,6,7,12", type=str)
    parser.add_argument("--group1_views", default="1,3,9,10,11", type=str)
    parser.add_argument("--prefixes", default="images_full,images_test", type=str)
    main(parser.parse_args())

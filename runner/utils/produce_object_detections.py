import argparse
import os
import json
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Path to input file with predictions")
    parser.add_argument("--output-path", type=str, help="Desired output path", default=None)
    args = parser.parse_args()

    if args.output_path in [None, ""]:
        args.output_path = args.input_path.rsplit(".", 1)[0] + "__object_detections.json"

    with open(args.input_path) as f:
        input_data = json.load(f)

    with open(args.output_path, "w") as f:
        json.dump(input_data["results"], f)

    print(f"Wrote {len(input_data['results'])} keys to {os.path.abspath(args.output_path)}")

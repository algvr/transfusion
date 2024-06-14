import argparse
import os
import json
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input-path", type=str, description="Path to input file")
    parser.add_argument("annotation-path", type=str, description="Path to directories with annotations")
    parser.add_argument("ttc-reference-path", type=str, description="Path to output of Ego4D SlowFast pipeline")
    parser.add_argument("output-path", type=str, description="Desired output path", default=None)
    args = parser.parse_args()

    if args.output_path in [None, ""]:
        args.output_path = args.input_path.rsplit(".", 1)[0] + "__adapted_ttc.json"

    with open(args.input_path) as f:
        all_object_detections = json.load(f)

    with open(args.ttc_reference_path) as f:
        ttc_preds = json.load(f)
    
    output_obj_detections = {}
    for k in tqdm([*all_object_detections.keys()]):
        output_obj_detections[k] = all_object_detections[k]
        for output_entry in output_obj_detections[k]:
            for i, prediction in enumerate(ttc_preds["results"][k]):
                if output_entry["score"] == prediction["score"]: 
                    if prediction["time_to_contact"] == []:
                        output_entry['time_to_contact'] = 0.5
                    else:
                        output_entry['time_to_contact'] = prediction["time_to_contact"]
                    # remove this entry because it was matched
                    ttc_preds["results"][k].pop(i)
                    break 

    output_obj_detections = {k: v for k, v in all_object_detections.items() if k in test_keys}
    output_preds = {
        "version": "1.0",
        "challenge": "ego4d_short_term_object_interaction_anticipation",
        "results": output_obj_detections
    }

    with open(args.output_path, "w") as f:
        json.dump(output_preds, f)

    print(f"Wrote {len(output_obj_detections)} keys to {os.path.abspath(args.output_path)}")

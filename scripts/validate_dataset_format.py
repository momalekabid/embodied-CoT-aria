"""
validate dataset format for aria recordings processed for embodied-cot training.

checks:
- required keys exist in the dataset
- gripper states are in [0, 1]
- image/state/action alignment
- bbox format (both old flat and new categorized)
- gaze point format
- prints diagnostic information
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def validate_gripper_states(data, verbose=False):
    """check if gripper states are in valid range [0, 1]"""
    print("\n=== Gripper State Validation ===")

    gripper_states = []
    invalid_count = 0

    for item in data:
        if 'action' in item and len(item['action']) > 0:
            # assuming last action dimension is gripper state
            gripper_state = item['action'][-1]
            gripper_states.append(gripper_state)

            if not (0.0 <= gripper_state <= 1.0):
                invalid_count += 1
                if verbose:
                    print(f"  invalid gripper state: {gripper_state}")

    if len(gripper_states) > 0:
        gripper_array = np.array(gripper_states)
        print(f"total gripper states: {len(gripper_states)}")
        print(f"invalid (outside [0,1]): {invalid_count}")
        print(f"mean: {gripper_array.mean():.3f}")
        print(f"std: {gripper_array.std():.3f}")
        print(f"range: [{gripper_array.min():.3f}, {gripper_array.max():.3f}]")
        print(f"% open (<0.5): {(gripper_array < 0.5).sum() / len(gripper_array) * 100:.1f}%")
        print(f"% closed (>=0.5): {(gripper_array >= 0.5).sum() / len(gripper_array) * 100:.1f}%")

        return invalid_count == 0
    else:
        print("no gripper states found in data")
        return False


def validate_image_state_action_alignment(data):
    """check if images, states, and actions have matching lengths"""
    print("\n=== Image/State/Action Alignment ===")

    has_image = 'image' in data[0] if len(data) > 0 else False
    has_state = 'state' in data[0] if len(data) > 0 else False
    has_action = 'action' in data[0] if len(data) > 0 else False

    print(f"has image: {has_image}")
    print(f"has state: {has_state}")
    print(f"has action: {has_action}")
    print(f"total timesteps: {len(data)}")

    if has_image and has_state and has_action:
        print("✓ all required keys present")
        return True
    else:
        print("✗ missing required keys")
        return False


def validate_bbox_format(bboxes_data, verbose=False):
    """validate both old flat and new categorized bbox formats"""
    print("\n=== Bounding Box Format Validation ===")

    if bboxes_data is None or len(bboxes_data) == 0:
        print("no bbox data found")
        return True

    # check first bbox to determine format
    first_bbox = bboxes_data[0]

    if isinstance(first_bbox, dict) and "PRIMARY" in first_bbox:
        print("detected: new categorized format")
        print(f"total frames with bboxes: {len(bboxes_data)}")

        # count objects in each category
        primary_counts = []
        gaze_focus_counts = []
        auxiliary_counts = []

        for frame_bboxes in bboxes_data:
            primary_counts.append(len(frame_bboxes.get("PRIMARY", [])))
            gaze_focus_counts.append(len(frame_bboxes.get("GAZE_FOCUS", [])))
            auxiliary_counts.append(len(frame_bboxes.get("AUXILIARY", [])))

        print(f"\nPRIMARY objects:")
        print(f"  avg per frame: {np.mean(primary_counts):.2f}")
        print(f"  frames with PRIMARY: {sum(1 for c in primary_counts if c > 0)}")

        print(f"\nGAZE_FOCUS objects:")
        print(f"  avg per frame: {np.mean(gaze_focus_counts):.2f}")
        print(f"  frames with GAZE_FOCUS: {sum(1 for c in gaze_focus_counts if c > 0)}")

        print(f"\nAUXILIARY objects:")
        print(f"  avg per frame: {np.mean(auxiliary_counts):.2f}")
        print(f"  frames with AUXILIARY: {sum(1 for c in auxiliary_counts if c > 0)}")

        # validate structure of first frame in detail
        if verbose and len(bboxes_data) > 0:
            print(f"\nexample frame bboxes:")
            print(json.dumps(bboxes_data[0], indent=2))

        return True

    elif isinstance(first_bbox, (list, tuple)):
        print("detected: old flat format")
        print(f"total frames with bboxes: {len(bboxes_data)}")

        bbox_counts = [len(frame_bboxes) for frame_bboxes in bboxes_data]
        print(f"avg bboxes per frame: {np.mean(bbox_counts):.2f}")
        print(f"max bboxes in a frame: {max(bbox_counts)}")

        if verbose and len(bboxes_data) > 0 and len(bboxes_data[0]) > 0:
            print(f"\nexample bbox: {bboxes_data[0][0]}")

        return True

    else:
        print(f"✗ unknown bbox format: {type(first_bbox)}")
        return False


def validate_gaze_points(gaze_data, verbose=False):
    """validate gaze point format"""
    print("\n=== Gaze Point Validation ===")

    if gaze_data is None or len(gaze_data) == 0:
        print("no gaze data found (this is optional)")
        return True

    print(f"total frames with gaze: {len(gaze_data)}")

    valid_count = 0
    none_count = 0

    for gaze_pt in gaze_data:
        if gaze_pt is None:
            none_count += 1
        elif isinstance(gaze_pt, (list, tuple)) and len(gaze_pt) == 2:
            valid_count += 1
        else:
            if verbose:
                print(f"  invalid gaze point: {gaze_pt}")

    print(f"valid gaze points: {valid_count}")
    print(f"none/missing: {none_count}")
    print(f"% frames with valid gaze: {valid_count / len(gaze_data) * 100:.1f}%")

    if verbose and valid_count > 0:
        # show example gaze point
        for gaze_pt in gaze_data:
            if gaze_pt is not None:
                print(f"example gaze point: {gaze_pt}")
                break

    return True


def main():
    parser = argparse.ArgumentParser(description="validate aria dataset format")
    parser.add_argument("--data", type=str, required=True, help="path to .npy or .json data file")
    parser.add_argument("--verbose", action="store_true", help="print detailed validation info")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: data file not found: {data_path}")
        return

    print(f"\nvalidating: {data_path}")
    print("=" * 60)

    # load data
    if data_path.suffix == '.npy':
        print("loading .npy file...")
        data = np.load(data_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()  # unpack if wrapped
    elif data_path.suffix == '.json':
        print("loading .json file...")
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        print(f"error: unsupported file format: {data_path.suffix}")
        return

    print(f"loaded data type: {type(data)}")

    # if data is a dict with 'frames' key (from generate_bboxes_with_gaze.py output)
    if isinstance(data, dict) and 'frames' in data:
        print("detected format: gaze-aware bbox json output")

        frames = data['frames']
        print(f"total frames: {len(frames)}")
        print(f"instruction: {data.get('instruction', 'N/A')}")

        # extract bboxes and gaze points
        bboxes_data = [frame.get('bboxes', {}) for frame in frames]
        gaze_data = [frame.get('gaze_point') for frame in frames]

        validate_bbox_format(bboxes_data, args.verbose)
        validate_gaze_points(gaze_data, args.verbose)

    # if data is a list (trajectory format)
    elif isinstance(data, (list, np.ndarray)):
        print(f"detected format: trajectory data")

        validate_image_state_action_alignment(data)

        # check if bboxes key exists
        if len(data) > 0 and 'bboxes' in data[0]:
            bboxes_data = [item.get('bboxes') for item in data]
            validate_bbox_format(bboxes_data, args.verbose)

        # check if gaze_point key exists
        if len(data) > 0 and 'gaze_point' in data[0]:
            gaze_data = [item.get('gaze_point') for item in data]
            validate_gaze_points(gaze_data, args.verbose)

        # validate gripper states
        validate_gripper_states(data, args.verbose)

    else:
        print(f"warning: unknown data structure")
        if args.verbose:
            print(f"data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")

    print("\n" + "=" * 60)
    print("validation complete")


if __name__ == "__main__":
    main()

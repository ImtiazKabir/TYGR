"""
View predictions from TYGR predict output.

Usage:
    ./TYGR view_predictions predictions.pkl [-v]
"""

from argparse import ArgumentParser
import pickle


def setup_parser(parser: ArgumentParser):
    parser.add_argument("predictions", type=str, help="Predictions pickle file from ./TYGR predict")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all details")
    parser.add_argument("--function", type=str, default=None, help="Filter by function address (hex)")


def main(args):
    with open(args.predictions, "rb") as f:
        var_dict = pickle.load(f)

    print(f"=== TYGR Type Predictions ===")
    print(f"Total functions: {len(var_dict)}\n")

    for func_addr, variables in var_dict.items():
        # Filter by function if specified
        if args.function:
            filter_addr = int(args.function, 16) if args.function.startswith("0x") else int(args.function)
            if func_addr != filter_addr:
                continue

        print(f"Function @ {hex(func_addr)}")
        print("-" * 40)

        for loc_key, predictions in variables.items():
            loc_type, loc_val = loc_key

            if loc_type == "cfa":
                loc_str = f"  Stack [CFA{loc_val:+d}]"
            elif loc_type == "reg":
                loc_str = f"  Register {loc_val}"
            elif loc_type == "addr":
                loc_str = f"  Address {hex(loc_val)}"
            else:
                loc_str = f"  {loc_type}: {loc_val}"

            # Get unique predicted types
            pred_types = set()
            for pred in predictions:
                if len(pred) >= 3:
                    _, _, btype = pred
                    pred_types.add(str(btype))

            types_str = ", ".join(sorted(pred_types)) if pred_types else "unknown"
            print(f"{loc_str}: {types_str}")

            if args.verbose:
                for pred in predictions:
                    if len(pred) >= 3:
                        _, pc, btype = pred
                        print(f"      @ PC {hex(pc)}: {btype}")

        print()

    if not var_dict:
        print("No predictions found in the file.")

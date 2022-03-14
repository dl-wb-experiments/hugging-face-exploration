import json
from collections import Counter

from constants import REPORTS_PATH


def process_results():
    with open(REPORTS_PATH / "cumulative_onnx_to_ir.json") as f:
        stat = json.load(f)
    with open(REPORTS_PATH / "string_replace.json") as f:
        replacements = json.load(f)

    results = []

    for model_name, error in stat["rejected"].items():
        flag = False

        for model, new_error in replacements["swap"].items():
            if model_name == model:
                results.append(new_error)
                flag = True

        if not flag:
            for substring, new_error in replacements["in"].items():
                if substring in error and not flag:
                    results.append(new_error)
                    flag = True
                    continue

        if not flag:
            for string, new_error in replacements["eq"].items():
                if error == string and not flag:
                    results.append(new_error)
                    flag = True
                    continue

        if not flag:
            results.append(error)

    return Counter(results)


def main():
    report = process_results()
    with open(REPORTS_PATH / "rejected_report.json", "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()

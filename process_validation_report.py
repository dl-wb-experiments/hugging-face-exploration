import json

from constants import ROOT_PATH


def main():
    local_report_dir_path = ROOT_PATH / 'reports' / 'small'
    final_report = ROOT_PATH / 'reports' / 'cumulative_onnx_to_ir.json'

    overall_status = {}
    for p in local_report_dir_path.rglob("*.json"):
        with open(p) as f:
            content = json.load(f)
        model_name = list(content.keys())[0]
        model_status = content[model_name]
        overall_status[model_name] = model_status

    success = []
    fails = {}
    for model_name, model_status in overall_status.items():
        if model_status == 'success':
            success.append(model_name)
            continue
        fails[model_name] = model_status

    report_content = {
        'summary': {
            'total': len(overall_status),
            'accepted': len(success),
            'rejected': len(fails)
        },
        'accepted': sorted(success),
        'rejected': {
            model_name: fails[model_name] for model_name in sorted(fails.keys())
        }
    }
    with open(final_report, 'w') as f:
        json.dump(report_content, f)


if __name__ == '__main__':
    main()

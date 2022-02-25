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
    reasons_summary = {}
    hf_error = 'Model cannot be downloaded from HF'
    for model_name, model_status in overall_status.items():
        if model_status == 'success':
            success.append(model_name)
            continue

        if hf_error == model_status:
            try:
                reasons_summary['hf_download_other (killed?)'] += 1
            except KeyError:
                reasons_summary['hf_download_other'] = 1
        elif hf_error in model_status:
            key = f'hf_download {model_status[len(hf_error) + 2:len(hf_error) + 2 + 120]}'
            try:
                reasons_summary[key] += 1
            except KeyError:
                reasons_summary[key] = 1
        else:
            try:
                reasons_summary[model_status] += 1
            except KeyError:
                reasons_summary[model_status] = 1

        fails[model_name] = model_status

    report_content = {
        'summary': {
            'total': len(overall_status),
            'accepted': len(success),
            'rejected': {
                'total': len(fails),
                'summary': {
                    reason: reasons_summary[reason] for reason in sorted(reasons_summary.keys())
                }
            }
        },
        'accepted': sorted(success),
        'rejected': {
            model_name: fails[model_name] for model_name in sorted(fails.keys())
        }
    }
    with open(final_report, 'w') as f:
        json.dump(report_content, f, indent=4)


if __name__ == '__main__':
    main()

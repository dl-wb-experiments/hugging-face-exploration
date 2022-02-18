import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from google.protobuf.json_format import MessageToDict

import onnx
import onnxruntime as rt

from constants import ROOT_PATH, REPORTS_PATH
from log_config import setup_logging


def run_console_tool(tool_path: Path, *args, **kwargs):
    python_executable = 'bash'

    kwargs_processed = []
    for item in kwargs.items():
        if item[0] == 'env':
            continue
        kwargs_processed.extend(map(str, item))

    options = [
        str(python_executable), str(tool_path),
        *args,
        *kwargs_processed
    ]

    if kwargs.get('env'):
        return subprocess.run(options, capture_output=True, env=kwargs.get('env'))
    return subprocess.run(options, capture_output=True)


def translate_inputs_to_openvino_mo_cli(inputs):
    shapes = []
    layouts = []
    names = []
    for input_name in inputs:
        names.append(input_name)
        roles = []
        for dim in inputs[input_name]:
            mapping = {
                'batch': 'n',
                'encoder_sequence': 'c',
                'decoder_sequence': 'c',
                'sequence': 'c'
            }
            roles.append(mapping.get(dim.get('layout_role')))

        are_roles_defined = None not in roles

        if not are_roles_defined:
            raise ValueError('Layouts are not defined for a model')
        else:
            role_str = f'{input_name}({"".join(roles)})'
        layouts.append(role_str)

        values = [dim.get('value') for dim in inputs[input_name]]
        are_values_defined = None not in values

        if not are_values_defined:
            if are_roles_defined:
                mapping_roles = {
                    'n': 1,
                    'c': 128
                }
                values = [str(mapping_roles[role]) for role in roles]
                shape_str = f'[{",".join(values)}]'
            else:
                raise ValueError('Shape is not defined! Cannot proceed!')
        else:
            shape_str = f'[{",".join(values)}]'
        shapes.append(shape_str)

    return ','.join(names), ','.join(shapes), ','.join(layouts)


class MOConversionError(Exception):
    pass


def run_openvino_mo_conversion(onnx_model_path, ir_model_path, inputs):
    tool_path = ROOT_PATH / 'scripts' / 'convert.sh'
    names, shapes, layouts = translate_inputs_to_openvino_mo_cli(inputs)
    env = dict(os.environ,
               **{
                   'INPUT_MODEL': str(onnx_model_path),
                   'INPUT_NAMES': f'{names}',
                   'INPUT_SHAPES': f'{shapes}',
                   'INPUT_LAYOUTS': f'{layouts}',
                   'OUTPUT_MODEL': str(ir_model_path)
               })

    res_process = run_console_tool(tool_path, env=env)
    logging.info(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    logging.info(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0:
        raise MOConversionError('Model cannot be converted')


def get_onnx_inputs(model_path):
    model = onnx.load_model(model_path)

    sess = rt.InferenceSession(str(model_path))
    inputs = sess.get_inputs()
    input_names = [i.name for i in inputs]

    inputs = {}
    for inp in model.graph.input:
        if inp.name not in input_names:
            continue

        inp_data = dict(MessageToDict(inp))
        dim_info = []
        for dim in inp_data.get('type').get('tensorType').get('shape').get('dim'):
            dim_info.append({
                'layout_role': dim.get('dimParam'),
                'value': dim.get("dimValue")
            })
        inputs[inp.name] = dim_info
    return inputs


def filter_onnx_inputs(inputs):
    return {
        key: value for key, value in inputs.items() if 'decoder' not in key
    }


class BenchmarkError(Exception):
    pass


def run_openvino_inference(ir_model):
    tool_path = ROOT_PATH / 'scripts' / 'benchmark.sh'
    env = dict(os.environ,
               **{
                   'IR_MODEL_XML': str(ir_model),
               })

    res_process = run_console_tool(tool_path, env=env)
    logging.info(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    logging.info(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0:
        raise BenchmarkError('Model cannot be benchmarked')


def check_single_model(onnx_path, ir_dir_path):
    inputs = get_onnx_inputs(onnx_path)
    real_inputs = inputs

    run_openvino_mo_conversion(onnx_path, ir_dir_path, real_inputs)

    ir_model_path = ir_dir_path / 'model.xml'
    run_openvino_inference(ir_model_path)


def download_convert_model_from_hf(model_name, onnx_path):
    tool_path = ROOT_PATH / 'scripts' / 'download.sh'
    env = dict(os.environ,
               **{
                   'HF_MODEL_NAME': model_name,
                   'ONNX_MODEL_PATH': str(onnx_path)
               })

    res_process = run_console_tool(tool_path, env=env)
    logging.info(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    logging.info(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0:
        raise ValueError('Model cannot be downloaded from HF')


def clean_resources(hf_model_name, onnx_dir_path, ir_model_dir_path):
    try:
        shutil.rmtree(onnx_dir_path)
    except OSError:
        logging.info(f'Unable to remove {onnx_dir_path}')

    try:
        shutil.rmtree(ir_model_dir_path)
    except OSError:
        logging.info(f'Unable to remove {onnx_dir_path}')

    path = Path.home() / '.cache' / 'huggingface' / 'transformers'
    model_hash = None
    for p in path.rglob("*.json"):
        with open(p) as f:
            content = json.load(f)
            if hf_model_name in content.get('url'):
                model_hash = p.stem
                break

    if model_hash is None:
        raise ValueError(f'Cannot find a downloaded model {hf_model_name}')

    for p in path.rglob(f"{model_hash}*"):
        logging.info(f'Removing {hf_model_name} model. File: {str(p)}')
        p.unlink()


def process_single_model(model_name, idx, clean=True):
    logging.info(f'{idx} Processing {model_name} ...')

    new_name = model_name.replace('/', '_')

    onnx_dir_path = ROOT_PATH / 'onnx_models' / new_name
    ir_model_path = ROOT_PATH / 'ir_models' / new_name
    onnx_dir_path.mkdir(exist_ok=True, parents=True)
    ir_model_path.mkdir(exist_ok=True, parents=True)

    msg = None
    try:
        download_convert_model_from_hf(model_name, onnx_dir_path)
    except ValueError:
        msg = 'Failed to download from Hugging Face'
        logging.info(f'{idx} {msg}')

    if not msg:
        onnx_path = onnx_dir_path / 'model.onnx'
        try:
            check_single_model(onnx_path, ir_model_path)
        except MOConversionError:
            msg = 'Failed to convert ONNX->IR'
            logging.info(f'{idx} {msg}')
        except BenchmarkError:
            msg = 'Failed to benchmark IR'
            logging.info(f'{idx} {msg}')

    if clean:
        clean_resources(model_name, onnx_dir_path, ir_model_path)

    if not msg:
        logging.info(f'{idx} OpenVINO success with {model_name}!')
        msg = 'success'

    local_report_path = ROOT_PATH / 'reports' / 'small' / new_name / 'report.json'
    local_report_path.mkdir(exist_ok=True, parents=True)
    with open(local_report_path) as f:
        json.dump({model_name: msg}, f)

    return model_name, msg


def get_accepted_models(report_path):
    with open(report_path) as f:
        content = json.load(f)
    return content.get('accepted')


def print_report(results):
    success_models = []
    failed_models = {}
    for model_name, result in results:
        if result == 'success':
            success_models.append(model_name)
            continue
        failed_models[model_name] = result

    logging.info(f'Failed models: {len(failed_models)}')
    for i in failed_models:
        logging.info(f'\t{i}')

    accepted_models = len(success_models)
    rejected_models = len(failed_models)
    total_models = len(success_models) + rejected_models

    logging.info(f'Total found models: {total_models}')
    logging.info(f'Rejected models: {rejected_models}')
    logging.info(f'Accepted models: {accepted_models}')

    result_json = {
        'summary': {
            'total': total_models,
            'accepted': accepted_models,
            'rejected': rejected_models,
        },
        'accepted': sorted(success_models),
        'rejected': {
            model_id: failed_models[model_id] for model_id in sorted(failed_models.keys())
        }
    }

    REPORTS_PATH.mkdir(exist_ok=True)
    result_report_path = REPORTS_PATH / 'onnx_to_ir_result_report.json'

    with open(result_report_path, 'w') as f:
        json.dump(result_json, f)


def main():
    setup_logging(log_filename='onnx_to_openvino.log')

    logging.info('Started')

    report_path = ROOT_PATH / 'reports' / 'hf_to_onnx.json'
    all_models_names = get_accepted_models(report_path)
    total_names = len(all_models_names)

    results = []
    for idx, model_name in enumerate(all_models_names[:2]):
        results.append(process_single_model(model_name, f'{idx + 1}/{total_names}'))

    print_report(results)


if __name__ == '__main__':
    main()

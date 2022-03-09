import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from google.protobuf.json_format import MessageToDict

import onnx
import onnxruntime as rt
from openvino.runtime import Core
from transformers import AutoTokenizer

from constants import ROOT_PATH, REPORTS_PATH
from log_config import setup_logging

core = Core()


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


class ONNXConversionError(Exception):
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


class ONNXIRToleranceError(Exception):
    pass


def check_logits_is_close(one, other):
    is_close = np.isclose(one, other, rtol=1e-04, atol=1e-04)  # 1e-04
    diff = np.abs(one - other)
    if not np.all(is_close):
        raise ONNXIRToleranceError(f"Max diff is {np.max(diff)}")


def get_ir_logits(ir_path, tokenized_data):
    compiled_model = core.compile_model(str(ir_path), device_name="CPU")
    res = compiled_model.infer_new_request(tokenized_data)
    return res[compiled_model.output()]


def get_onnx_logits(onnx_path, tokenized_data):
    session = rt.InferenceSession(str(onnx_path))
    return session.run(None, tokenized_data)[0]


def check_single_model(onnx_path, ir_dir_path, tokenizer):
    try:
        inputs = get_onnx_inputs(onnx_path)
    except FileNotFoundError:
        raise ONNXConversionError('Model cannot be converted to ONNX')

    run_openvino_mo_conversion(onnx_path, ir_dir_path, inputs)

    ir_model_path = ir_dir_path / 'model.xml'
    run_openvino_inference(ir_model_path)

    tokenized_text = tokenizer(
        "test text, this is a test texts, deal with it!!!",
        padding=True,
        truncation=True,
        pad_to_multiple_of=128,
        max_length=128,
        return_tensors="np",
    )
    tokenized_text = {name: np.atleast_2d(value) for name, value in tokenized_text.items()}

    ir_logits = get_ir_logits(ir_model_path, tokenized_text)
    onnx_logits = get_onnx_logits(onnx_path, tokenized_text)

    check_logits_is_close(ir_logits, onnx_logits)


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
        all_exception_lines = str(res_process.stderr.decode("utf-8")).split('\n')
        r = re.compile('.*error.*', re.IGNORECASE)
        exception_lines = list(filter(r.match, all_exception_lines))

        if not exception_lines:
            msg = 'Model cannot be downloaded from HF'
        else:
            line = exception_lines[-1]
            if 'onnxruntime.' in line:
                reason = line
            else:
                reason = line.split('.')[0]
            msg = 'Model cannot be downloaded from HF: ' + reason
        raise ValueError(msg)


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
    model_hashes = []
    for p in path.rglob("*.json"):
        try:
            with open(p) as f:
                content = json.load(f)
                if hf_model_name in content.get('url'):
                    model_hashes.append(p.stem)
        except FileNotFoundError:
            logging.info(f'Unable to check {p} - it was already removed')

    if not model_hashes:
        logging.info(f'Cannot find a downloaded model {hf_model_name}')
        return

    for model_hash in model_hashes:
        for p in path.rglob(f"{model_hash}*"):
            logging.info(f'Removing {hf_model_name} model. File: {str(p)}')
            try:
                p.unlink()
            except FileNotFoundError:
                logging.info(f'Unable to remove {p} - it was already removed')


def process_single_model(model_name, idx, clean=True):
    logging.info(f'{idx} Processing {model_name} ...')

    new_name = model_name.replace('/', '_')

    local_report_dir_path = ROOT_PATH / 'reports' / 'small' / new_name
    local_report_dir_path.mkdir(exist_ok=True, parents=True)

    local_report_path = local_report_dir_path / 'report.json'

    try:
        if local_report_path.exists():
            logging.info(f'{idx} Already processed model {model_name}. Checking report...')
            try:
                with open(local_report_path) as f:
                    content = json.load(f)
                    msg = content[model_name]
            except json.decoder.JSONDecodeError:
                logging.info(f'{idx} Previous processing report for {model_name} is broken')
                local_report_path.unlink()
            else:
                logging.info(f'{idx} Already processed model {model_name}. Skipping...')
                return model_name, msg

        onnx_dir_path = ROOT_PATH / 'onnx_models' / new_name
        ir_model_path = ROOT_PATH / 'ir_models' / new_name
        onnx_dir_path.mkdir(exist_ok=True, parents=True)
        ir_model_path.mkdir(exist_ok=True, parents=True)

        msg = None
        try:
            download_convert_model_from_hf(model_name, onnx_dir_path)
        except ValueError as e:
            msg = str(e)
            logging.info(f'{idx} {msg}')

        if not msg:
            onnx_path = onnx_dir_path / 'model.onnx'
            try:
                check_single_model(
                    onnx_path, ir_model_path, AutoTokenizer.from_pretrained(model_name)
                )
            except ONNXIRToleranceError:
                msg = f"Failed to convert ONNX->IR: Max diff"
            except MOConversionError as e:
                msg = f'Failed to convert ONNX->IR: {e}'
                logging.info(f'{idx} {msg}')
            except BenchmarkError as e:
                msg = f'Failed to benchmark IR: {e}'
                logging.info(f'{idx} {msg}')
            except ONNXConversionError as e:
                msg = f'Model cannot be converted to ONNX: {e}'
                logging.info(f'{idx} {msg}')
            except RuntimeError as e:
                msg = f'OpenVINO Error: {e}'
                logging.info(f'{idx} {msg}')
            except Exception as e:
                msg = f'{model_name} Unexpected Exception: {e}'
                logging.info(f'{idx} {msg}')

        if clean:
            clean_resources(model_name, onnx_dir_path, ir_model_path)

        if not msg:
            logging.info(f'{idx} OpenVINO success with {model_name}!')
            msg = 'success'
    except Exception as e:
        msg = f'General error: {e}'

    with open(local_report_path, 'w') as f:
        json.dump({model_name: msg}, f, indent=4)

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
        json.dump(result_json, f, indent=4)


def main():
    setup_logging(log_filename='onnx_to_openvino.log')

    logging.info('Started')

    report_path = ROOT_PATH / 'reports' / 'hf_to_onnx.json'
    all_models_names = get_accepted_models(report_path)
    total_names = len(all_models_names)

    all_models_names = [
        'Emirhan/51k-finetuned-bert-model',
        'Elluran/Hate_speech_detector',
        'Emanuel/bertweet-emotion-base',
        'Fengkai/distilbert-base-uncased-finetuned-emotion'
    ]

    results = []
    for idx, model_name in enumerate(all_models_names[:5]):
        results.append(process_single_model(model_name, f'{idx + 1}/{total_names}', clean=False))

    print_report(results)


if __name__ == '__main__':
    main()

import json
import os
import shutil
import subprocess
from pathlib import Path

from google.protobuf.json_format import MessageToDict

import onnx
import onnxruntime as rt

from constants import ROOT_PATH


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
    print(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    print(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

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
    print(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    print(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0:
        raise BenchmarkError('Model cannot be benchmarked')


def check_single_model(onnx_path, ir_dir_path):
    inputs = get_onnx_inputs(onnx_path)
    real_inputs = inputs

    try:
        run_openvino_mo_conversion(onnx_path, ir_dir_path, real_inputs)
    except MOConversionError:
        print('Model failed to convert to OpenVINO IR')

    ir_model_path = ir_dir_path / 'model.xml'
    try:
        run_openvino_inference(ir_model_path)
    except BenchmarkError:
        print('Model failed to infer with OpenVINO')


def download_convert_model_from_hf(model_name, onnx_path):
    tool_path = ROOT_PATH / 'scripts' / 'download.sh'
    env = dict(os.environ,
               **{
                   'HF_MODEL_NAME': model_name,
                   'ONNX_MODEL_PATH': str(onnx_path)
               })

    res_process = run_console_tool(tool_path, env=env)
    print(f'SUBPROCESS: {str(res_process.stdout.decode("utf-8"))}')
    print(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0:
        raise ValueError('Model cannot be downloaded from HF')


def clean_resources(hf_model_name, onnx_dir_path, ir_model_dir_path):
    shutil.rmtree(onnx_dir_path)
    shutil.rmtree(ir_model_dir_path)

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
        print(f'Removing {hf_model_name} model. File: {str(p)}')
        p.unlink()


def process_single_model(model_name):
    new_name = model_name.replace('/', '_')

    onnx_dir_path = ROOT_PATH / 'onnx_models' / new_name
    ir_model_path = ROOT_PATH / 'ir_models' / new_name
    onnx_dir_path.mkdir(exist_ok=True, parents=True)
    ir_model_path.mkdir(exist_ok=True, parents=True)

    download_convert_model_from_hf(model_name, onnx_dir_path)

    onnx_path = onnx_dir_path / 'model.onnx'
    check_single_model(onnx_path, ir_model_path)

    clean_resources(model_name, onnx_dir_path, ir_model_path)


def main():
    model_name = 'prajjwal1/bert-tiny-mnli'
    process_single_model(model_name)


if __name__ == '__main__':
    main()

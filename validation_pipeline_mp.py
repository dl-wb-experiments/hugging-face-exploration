import logging
import multiprocessing as mp

from constants import ROOT_PATH
from log_config import setup_logging
from validation_pipeline import get_accepted_models, process_single_model, print_report


def main():
    setup_logging(log_filename='onnx_to_openvino_mp.log')

    logging.info('Started')

    num_cores = mp.cpu_count()
    logging.info(f'Multiprocessing enabled for {num_cores} cores')

    pool = mp.Pool(num_cores)

    report_path = ROOT_PATH / 'reports' / 'hf_to_onnx.json'
    all_models_names = get_accepted_models(report_path)
    total_names = len(all_models_names)

    result_objects = [pool.apply_async(process_single_model, args=(model_name, f'{idx + 1}/{total_names}', True))
                      for idx, model_name in enumerate(all_models_names[:4])]
    results = [r.get() for r in result_objects]

    print_report(results)


if __name__ == '__main__':
    main()

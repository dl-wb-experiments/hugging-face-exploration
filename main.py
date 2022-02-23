import logging

import json

from huggingface_hub import HfApi, ModelSearchArguments
from transformers.onnx import FeaturesManager

from constants import REPORTS_PATH
from log_config import setup_logging


def get_model_candidates_from_hf():
    api = HfApi()
    model_args = ModelSearchArguments()

    tags = (
        model_args.pipeline_tag.TextClassification,
        model_args.library.PyTorch,
        model_args.library.Transformers,
    )

    models = api.list_models(filter=tags, limit=10000, fetch_config=True, full=False)

    filtered_models = [model for model in models if is_transformer_encoder(model)]

    logging.info(f"Fetched {len(models)} models")
    logging.info(f"Filtered {len(models) - len(filtered_models)}")
    logging.info(f"No. of models after filtering: {len(filtered_models)}")

    return filtered_models, tags


def process_batch(models, start=None, end=None):
    no_config_models = []
    not_exportable_models = {}
    missed_type = []

    start = start or 0
    end = end or len(models)

    for idx, model in enumerate(models[start:end]):
        model_idx = idx + start + 1

        if model.config is None:
            no_config_models.append(model.modelId)
            logging.info(f'{model_idx} No config {model.modelId}')
            continue

        try:
            model_type = model.config["model_type"].replace("_", "-")
        except AttributeError:
            logging.info(f'{model_idx} No type in config {model.modelId}')
            missed_type.append(model.modelId)
            continue

        try:
            FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model.modelId)
        except KeyError:
            logging.info(f'{model_idx} Not supported for export {model.modelId}')
            not_exportable_models[model.modelId] = model_type
            continue

        logging.info(f'{model_idx} Accepting {model.modelId}')
    return no_config_models, not_exportable_models, missed_type


def print_report(no_config_models, not_exportable_models, missed_type, models, tags):
    logging.info(f'No config: {len(no_config_models)}')
    for i in no_config_models:
        logging.info(f'\t{i}')

    logging.info(f'Missed type: {len(missed_type)}')
    for i in missed_type:
        logging.info(f'\t{i}')

    logging.info(f'Not supported: {len(not_exportable_models)}')
    for model_id, model_type in not_exportable_models.items():
        logging.info(f'\t{model_id}\t\t{model_type}')

    all_rejected = {}
    for model_id, model_type in not_exportable_models.items():
        all_rejected[model_id] = {
            'reason': 'Unsupported model type',
            'model_type': model_type
        }
    for model_id in missed_type:
        all_rejected[model_id] = {
            'reason': 'No config file',
            'model_type': 'unknown'
        }

    total_models = len(models)
    rejected_models = len(all_rejected)
    accepted_models = total_models - rejected_models
    logging.info(f'Total found models: {total_models}')
    logging.info(f'Rejected models: {rejected_models}')
    logging.info(f'Accepted models: {accepted_models}')

    result_json = {
        'summary': {
            'tags': tags,
            'total': total_models,
            'accepted': accepted_models,
            'rejected': rejected_models,
        },
        'accepted': sorted([m.modelId for m in models if m.modelId not in all_rejected]),
        'rejected': {
            model_id: all_rejected[model_id] for model_id in sorted(all_rejected.keys())
        }
    }

    REPORTS_PATH.mkdir(exist_ok=True)
    result_report_path = REPORTS_PATH / 'hf_to_onnx.json'

    with open(result_report_path, 'w') as f:
        json.dump(result_json, f, indent=4)


def is_transformer_encoder(model):
    supported_model_types = {
        model_type for model_type, tasks in FeaturesManager._SUPPORTED_MODEL_TYPE.items()
        if not any("with-past" in task for task in tasks)
    }
    return (
            model.config is not None
            and "model_type" in model.config
            and model.config["model_type"] in supported_model_types
            and "architectures" in model.config
            and any("SequenceClassification" in arch for arch in model.config["architectures"])
    )


def main():
    setup_logging()

    logging.info('Started')

    models, tags = get_model_candidates_from_hf()

    no_config_models, not_exportable_models, missed_type = process_batch(models, end=26)

    print_report(no_config_models, not_exportable_models, missed_type, models, tags)


if __name__ == '__main__':
    main()

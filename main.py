import logging

import json
from pathlib import Path

from huggingface_hub import HfApi, ModelSearchArguments, cached_download
from huggingface_hub import hf_hub_url
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils.endpoint_helpers import AttributeDictionary
from transformers.onnx import FeaturesManager


def has_config(model: ModelInfo):
    for sibling in model.siblings:
        if sibling.rfilename == 'config.json':
            return True
    return False


def main():
    logging.basicConfig(filename='check_hf.log',
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Started')

    api = HfApi()
    model_args = ModelSearchArguments()

    models = api.list_models(filter=(
        model_args.pipeline_tag.TextClassification,
        # model_args.dataset.glue,
        model_args.library.PyTorch),
        limit=10000
    )

    no_config_models = []
    not_exportable_models = {}
    missed_type = []
    for idx, model in enumerate(models):
        if not has_config(model):
            logging.info(idx + 1, 'No config!!! ', model.modelId)
            no_config_models.append(model.modelId)
            continue

        hf_readme_url = hf_hub_url(repo_id=model.modelId, filename="config.json")
        readme_path = Path(cached_download(hf_readme_url))
        with open(readme_path) as f:
            model.config = AttributeDictionary(json.load(f))

        try:
            model_type = model.config.model_type.replace("_", "-")
        except AttributeError:
            logging.info(f'{idx + 1} No type in config {model.modelId}')
            missed_type.append(model.modelId)
            continue

        try:
            FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model.modelId)
        except KeyError:
            logging.info(f'{idx + 1} Not supported for export {model.modelId}')
            not_exportable_models[model.modelId] = model_type
            continue

        logging.info(f'{idx + 1} Skipping... {model.modelId}')

    logging.info(f'No config: {len(no_config_models)}')
    for i in no_config_models:
        logging.info(f'\t{i}')

    logging.info(f'Missed type: {len(missed_type)}')
    for i in missed_type:
        logging.info(f'\t{i}')

    logging.info(f'Not supported: {len(no_config_models)}')
    for model_id, model_type in not_exportable_models.items():
        logging.info(f'\t{model_id}\t\t{model_type}')

    total_models = len(models)
    rejected_models = len(list(not_exportable_models.keys())) + len(missed_type)
    accepted_models = total_models - rejected_models
    logging.info(f'Total found models: {total_models}')
    logging.info(f'Rejected models: {rejected_models}')
    logging.info(f'Accepted models: {accepted_models}')


if __name__ == '__main__':
    main()

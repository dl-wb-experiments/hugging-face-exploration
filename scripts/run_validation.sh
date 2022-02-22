source venv/bin/activate

rm -rf logs
mkdir -p logs

export TOKENIZERS_PARALLELISM=false

nohup python -u validation_pipeline_mp.py \
                > logs/validation_pipeline_mp.log \
                2> logs/validation_pipeline_mp_err.log &

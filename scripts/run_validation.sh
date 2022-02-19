source venv/bin/activate

nohup python -u validation_pipeline_mp.py \
                > logs/validation_pipeline_mp.log \
                2> validation_pipeline_mperr.log &

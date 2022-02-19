source venv/bin/activate

rm -rf logs

nohup python -u validation_pipeline_mp.py \
                > logs/validation_pipeline_mp.log \
                2> logs/validation_pipeline_mp_err.log &

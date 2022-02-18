# make the script verbose
set -x

source venv/bin/activate

#IR_MODEL_XML=./ir_model2/model.xml

benchmark_app \
    -m ${IR_MODEL_XML} \
    -b 1 -nstreams 1 \
    -d CPU -t 2
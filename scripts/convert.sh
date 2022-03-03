# make the script verbose
set -x

source venv/bin/activate

#INPUT_MODEL=./onnx/model.onnx
#INPUT_NAMES='input_ids,attention_mask'
#INPUT_SHAPES='[1,128],[1,128]'
#INPUT_LAYOUTS='input_ids(nc),attention_mask(nc)'
#OUTPUT_MODEL=./ir_model

python -m openvino.tools.mo \
    --data_type 'FP32' \
    --input ${INPUT_NAMES} \
    --input_shape ${INPUT_SHAPES} \
    --input_model ${INPUT_MODEL} \
    --model_name 'model' \
    --framework 'onnx' \
    --output_dir ${OUTPUT_MODEL} \
    --progress \
    --layout ${INPUT_LAYOUTS}
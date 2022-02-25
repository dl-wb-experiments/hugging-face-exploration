# make the script verbose
set -x

source venv/bin/activate

#HF_MODEL_NAME=lincoln/flaubert-mlsum-topic-classification
#ONNX_MODEL_PATH=onnx_ph/
#Check for this type of output from transformers.onnx: Some weights of the model checkpoint at katrin-kc/bert-finetuned-imdb were not used when initializing BertModel: ['classifier.bias', 'classifier.weight']

python -m transformers.onnx \
          --model=${HF_MODEL_NAME} \
          --feature=sequence-classification \
          --atol 1e-4 \
          ${ONNX_MODEL_PATH}

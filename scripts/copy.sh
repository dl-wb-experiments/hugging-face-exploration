# make the script verbose
set -x

WHERE_TO_KEEP=~/Downloads/new_report/
#WHAT_TO_KEEP=/root/hugging-face-exploration/logs/onnx_to_openvino_mp.log
WHAT_TO_KEEP=/root/hugging-face-exploration/reports/cumulative_onnx_to_ir.json
SERVER_IP=remote-alexander

#rm -rf ${WHERE_TO_KEEP}
#mkdir -p ${WHERE_TO_KEEP}

scp -i ~/.ssh/id_rsa_ya -r remote-alexander:${WHAT_TO_KEEP} ${WHERE_TO_KEEP}

# make the script verbose
set -e

WHERE_TO_KEEP=~/Downloads/new_report/
SERVER_IP=remote-alexander
DO_CLEANUP=1

if [ "${DO_CLEANUP}" == "1" ]; then
  echo "Cleaning the system ..."
  rm -rf ${WHERE_TO_KEEP}
  mkdir -p ${WHERE_TO_KEEP}
fi


FILES_TO_COPY=(
  '/root/hugging-face-exploration/logs/onnx_to_openvino_mp.log'
  '/root/hugging-face-exploration/reports/cumulative_onnx_to_ir.json'
  '/root/hugging-face-exploration/reports/hf_to_onnx.json'
)

for remote_file_path in "${FILES_TO_COPY[@]}"
do
  echo "downloading ${remote_file_path} ..."
  scp -i ~/.ssh/id_rsa_ya -r remote-alexander:${remote_file_path} ${WHERE_TO_KEEP}
done

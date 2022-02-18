#apt -y update
#apt -y upgrade
#apt install -y git
#git clone https://github.com/dl-wb-experiments/hugging-face-exploration
#cd hugging-face-exploration
#bash scripts/bootstrap.sh

apt -y update
apt -y upgrade
apt install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        htop

python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate

python -m pip install -r requirements.txt

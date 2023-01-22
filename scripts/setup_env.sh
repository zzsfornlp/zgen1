#

# prepare the new env
conda create -y -n zgen0 python=3.8
. "`conda info --base`/etc/profile.d/conda.sh"
conda activate zgen0
conda install -y pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -y numpy pip scipy=1.6.2 pandas=1.2.5
pip install transformers==3.1.0
# bleu & rouge
pip install sacremoses==0.0.45
pip install sacrebleu==1.5.1
pip install rouge-score==0.0.4
# nvcc
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
bash ./cuda_11.1.0_455.23.05_linux.run --silent --toolkit --installpath=`pwd`/cuda
# apex
git clone https://github.com/NVIDIA/apex
cd apex; git checkout ebcd7f084bba96bdb0c3fdf396c3c6b02e745042; CUDA_HOME=`pwd`/../cuda pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./; cd ..
# --

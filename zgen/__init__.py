#

# based on msp2 (specifically simplified for this project)

# requirements
# python=3.8
# conda install pytorch=1.5.0 -c pytorch
# pip install transformers==3.1.0
# conda install scipy pandas
# apex: see https://github.com/nvidia/apex
# the one that we use: 00c1e56d02f2987e71ee40417ea7e938e1493475 or ebcd7f084bba96bdb0c3fdf396c3c6b02e745042
# note: for apex, need to install cudatoolkit (no driver) for nvcc and then set CUDA_HOME!
# --
# wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
# bash ./cuda_10.2.89_440.33.01_linux.run --toolkit --installpath=??/cuda
# bleu & rouge
# pip install sacremoses==0.0.45
# pip install sacrebleu==1.5.1
# pip install rouge-score==0.0.4
# --

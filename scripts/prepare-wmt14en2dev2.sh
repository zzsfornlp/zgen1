# --

mkdir data
cd data

wget http://dl.fbaipublicfiles.com/nat/original_dataset.zip
wget http://dl.fbaipublicfiles.com/nat/distill_dataset.zip
unzip original_dataset.zip
unzip distill_dataset.zip

# rename
for cl in en de; do
  cp wmt14_ende/train.en-de.${cl} train.${cl}
  cp wmt14_ende_distill/train.en-de.${cl} trainD.${cl}
  cp wmt14_ende/valid.en-de.${cl} dev.${cl}
  cp wmt14_ende/test.en-de.${cl} test.${cl}
done

# split for multi-gpu training
python3 ../src/scripts/shuffle_and_split.py input_paths:train.en,train.de,trainD.en,trainD.de output_paths:wmt14en2de.train.en,wmt14en2de.train.de,wmt14en2de.train.enD,wmt14en2de.train.deD
for wset in dev test; do
for cl in en de; do
cp ${wset}.${cl} wmt14en2de.${wset}.${cl}
done
done

# convert vocab
mkdir -p voc_wmt14en2de
python3 ../src/scripts/convert_vocab.py wmt14_ende/dict.en.txt voc_wmt14en2de/v_enc.pkl
python3 ../src/scripts/convert_vocab.py wmt14_ende/dict.de.txt voc_wmt14en2de/v_slm.pkl
cp voc_wmt14en2de/v_slm.pkl voc_wmt14en2de/v_ilm.pkl

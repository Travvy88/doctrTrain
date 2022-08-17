rm -rf doctr
git clone https://github.com/mindee/doctr

cp fixed_files/train_pytorch.py doctr/references/detection/train_pytorch.py
cp fixed_files/detection.py doctr/doctr/datasets/detection.py
# cp fixed_files/bbox_utils.py /usr/local/lib/python3.6/dist-packages/albumentations/core

rm -rf data
python3 dataset/dataset_for_training.py  https://at.ispras.ru/owncloud/index.php/s/mkhtvLDHE0mLU5F/download --augment

# Grad-PU and PUGAN

## Installation

* Install the following packages

```
pip install open3d einops scikit-learn tqdm h5py torch ninja
```

* Install the built-in libraries

```
cd models/Chamfer3D
python setup.py install
cd ../pointops
python setup.py install
```

#### dataset
We use the PU-Net dataset for training, you can refer to https://github.com/yulequan/PU-Net to download the .h5 dataset file, which can be directly used in this project.
#### modify some setting in the option/train_option.py
change opt['project_dir'] to where this project is located, and change opt['dataset_dir'] to where you store the dataset.
<br/>
also change params['train_split'] and params['test_split'] to where you save the train/test split txt files.
#### training
```
cd train
python train.py --exp_name=the_project_name --gpu=gpu_number --use_gan --batch_size=12
```


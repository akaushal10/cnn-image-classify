# CS6910 Assignment 2 Part A

[Link to Weights & Biases Report](https://wandb.ai/cs23m007/dl-assignment-2/reports/CS6910-Assignment-2--Vmlldzo3Mzk2NDcz)

## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies. requirements.txt file present in the root Repo.

```
pip install requirements.txt
python train.py
```

The number of filters, size of filters and activation function in each layer can be changed and  the number of neurons in the dense layer can be changed by passing command line arguments to the training script

```
python train.py --num_filters 32 --filter_size 256 --activation relu --neurons_dense 128
``` 

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `dp`, `dataset_path` | /kaggle/input/nature-12k/inaturalist_12K/ | Path where your inaturalist_12K dataset store
| `-e`, `--epochs` | 5 | Number of epochs to train neural network.[10, 15, 20 , 25 , 30] |
| `-b`, `--batch_size` | 16 | Batch size used to train neural network, choices: [16,32,64] | 
| `-bn`, `--batch_norm` | True | choices:  [True, False] | 
| `-da`, `--data_aug` | False | choices:  [True, False] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters, choices: [0.001,0.0001,0.0003,0.0005] | 
| `-nf`, `--num_filters` | 16 | Number of filters used in convolutional layers. | 
| `-fs`, `--filter_size` | [11,9,7,5,3] | choices: [ '7,5,5,3,3', '11,9,7,5,3'] | 
| `-a`, `--activation` | ReLU | choices:  ["ReLU", "LeakyReLU", "GELU", "SiLU", "Mish","ELU"] |
| `-sz`, `--neurons_dense` | 32 | NUmber of hidden neurons in dense layer choice: [32,64,128,256,512,1024 ]. |

## Examples, Usage and More

### Defining a Model

```python

model = Model(image_shape= image_shape,
                dropout= 0.1,
                activation= ReLU,
                batch_norm= True,
                size_filters= [11,9,7,5,3],
                filter_organization= 2,
                number_filters= 32,
                neurons_in_dense_layer= 512,
                num_conv_layers= 5
            ).to(device)

y_pred = model.train()
```

You can view the code by clicking this link: [Questions_1-4](<DL_ASSIGNMENT_2_PART_A.ipynb>)

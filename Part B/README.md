# CS6910 Assignment 2 Part A

[Link to Weights & Biases Report](https://wandb.ai/cs23m007/dl-assignment-2/reports/CS6910-Assignment-2--Vmlldzo3Mzk2NDcz)

## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies. requirements.txt file present in the root Repo.

```
pip install requirements.txt
python train.py
```

The model, number of layers to freeze, pretrained can be changed by passing command line arguments to the training script


```
python train.py --epochs 10 --batch_size 16
``` 

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-e`, `--epochs` | 5 | Number of epochs to train neural network.[10, 15, 20 , 25 , 30] |
| `-b`, `--batch_size` | 16 | Batch size used to train neural network, choices: [16,32,64] | 
| `dp`, `dataset_path` | /kaggle/input/nature-12k/inaturalist_12K/ | Path where your inaturalist_12K dataset store | 
| `-da`, `--data_aug` | False | choices:  [True, False] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters, choices: [0.001,0.0001,0.0003,0.0005] | 
| `-fk`, `--freeze_k` | 45 | Number of layers to freeze in the network |

## Examples, Usage and More

### Defining a Model

```python
# This is the model provided by torchvision
model = torchvision.models.inception_v3(pretrained=True),
 ```

You can view the code by clicking this link: [Questions_1-4](<DL_ASSIGNMENT_2_PART_B.ipynb>)

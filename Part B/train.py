import cv2
import glob
import random
import numpy as np
import torch
from pandas.core.common import flatten
torch.manual_seed(7)
torch.cuda.empty_cache()
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch import nn,optim
import torch.nn.functional as F
from tqdm import tqdm 
import wandb
import argparse


# constants
IMG_MODE = 'RGB'
TRAIN_LABEL = 'train'
TEST_LABEL = 'test'

# activation function
RELU_KEY = 'ReLU'
LEAKY_RELU_KEY = 'LeakyReLU'
GELU_KEY = 'GELU'
SILU_KEY = 'SiLU'
MISH_KEY = 'Mish'
ELU_KEY = 'ELU'

# wandb constants
WANDB_PROJECT_NAME="dl-assignment-2"
WANDB_ENTITY_NAME="cs23m007"

# wandb sweep param labels
NUMBER_FILTER_KEY = "number_filters"
ACTIVATION_FUNCTION_KEY = "activation"
FILTER_ORGANIZATION_KEY = "filter_organization"
DATA_AUGMENTATION_KEY = "data_aug"
BATCH_NORMALIZATION_KEY = "batch_norm"
DROPOUT_KEY = "dropout"
BATCH_SIZE_KEY = "batch_size"
EPOCHS_KEY = "epochs"
LEARNING_RATE_KEY = "learning_rate"
SIZE_FILTER_KEY = "size_filters"
DENSE_LAYER_NEURONS_KEY = "neurons_in_dense_layer"
PRETRAINED_KEY = "pretrained"
FREEZE_KEY = "freeze"

# wandb plot titles
TRAIN_ACCURACY_TITLE = "train_acc"
VALIDATION_ACCURACY_TITLE = "val_acc"
TEST_ACCURACY_TITLE = "test_acc"
TRAIN_LOSS_TITLE = "train_loss"
VALIDATION_LOSS_TITLE = "val_loss"
TEST_LOSS_TITLE = "test_loss"

# Ratio to split train and validation 0.8 means 80% train data and 20% validation data
TRAIN_DATASET_SPLIT_RATIO = 0.8

parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Project name used to track experiments in Weights & Biases dashboard",default=WANDB_PROJECT_NAME)
parser.add_argument("-we","--wandb_entity",help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default=WANDB_ENTITY_NAME)
parser.add_argument("-dp","--dataset_path",help="Path of folder where dataset located",default="/kaggle/input/nature-12k/inaturalist_12K/")
parser.add_argument("-e","--epochs",help="Number of epochs to train neural network.",choices=['5','10','15','20','25','30'],default=10)
parser.add_argument("-b","--batch_size",help="Batch size used to train neural network.",choices=['16','32','64'],default=64)
parser.add_argument("-lr","--learning_rate",help="Learning rate used to optimize model parameters",choices=['1e-3','1e-4'],default=0.001)
parser.add_argument("-da","--data_aug",help="Do you want Data Augumenation, 1 means Yes, 0 means No",choices=['1','0'],default=False)
parser.add_argument("-f","--freeze_k",help="Number of layer want to freeze, -1 if not and < 45",default=-1)

args = parser.parse_args()

DATASET_PATH = args.dataset_path
TEST_DATA_PATH = f'{DATASET_PATH}val/'
TRAIN_DATA_PATH = f'{DATASET_PATH}train/'

class DotDict:
    """
    Used to convert dict to an object
    """
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def convertIntoPercentage(x,n,digit=4):
    return round((x / n) * 100, digit)

def evaluate(device, loader, model):
    """
    Evaluate the performance of a neural network model on a dataset.

    Parameters:
        device (torch.device): The device to run the evaluation on (e.g., CPU or GPU).
        loader (torch.utils.data.DataLoader): DataLoader for loading batches of data.
        model (torch.nn.Module): The neural network model to evaluate.

    Returns:
        Tuple[float, float]: Accuracy and average loss of the model on the dataset.
    """

    # Initialize variables to keep track of correct predictions and total samples
    Y_cap_num,N_val = 0,0
    loss = 0
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation since no training is done during evaluation
    with torch.no_grad():
        for X, Y in tqdm(loader, total=len(loader)):
            X,Y = X.to(device=device),Y.to(device=device)
            
            # Forward pass: compute predicted outputs by passing inputs through the model
            Y_cap = model(X)
            loss += nn.CrossEntropyLoss()(Y_cap, Y).item()

            _, predictions = Y_cap.max(1)

            N_val = N_val + predictions.size(0)
            
            Y_cap_num = Y_cap_num +  (predictions == Y).sum().item()

    # Calculate accuracy and average loss
    acc = convertIntoPercentage(Y_cap_num , N_val)
    loss = loss/N_val
    return acc, loss

def freeze_layers(model, freeze_k):
    """
    Freeze layers in a PyTorch model up to a certain depth.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose layers are to be frozen.
        freeze_k (int): The index of the last layer to freeze. If freeze_k is -1,
                       all layers will be frozen.

    Returns:
        None
    """

    # If freeze_k is -1, freeze all layers
    if freeze_k == -1: 
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Initialize a counter to keep track of the layers
        k = 0

        # Iterate over model parameters
        for param in model.parameters():
            k += 1
            param.requires_grad = False
            
            # If reached the specified depth, stop freezing layers
            if k > freeze_k:
                return

class iNaturalist(Dataset):
    """
    Custom dataset class for iNaturalist dataset.

    Parameters:
        image_paths (list): List of file paths to images.
        class_to_idx (dict): Dictionary mapping class names to indices.
        transform (callable): A function/transform to apply to the images.
    """
    def __init__(self, image_paths, class_to_idx, transform):
        """
        Initialize the dataset with image paths, class mappings, and transformation.

        Args:
            image_paths (list): List of file paths to images.
            class_to_idx (dict): Dictionary mapping class names to indices.
            transform (callable): A function/transform to apply to the images.
        """
        self.all_images = image_paths
        self.current_transform = transform
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.all_images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """

        image_filepath = self.all_images[idx]

        # Read the image using OpenCV and convert color from BGR to RGB
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract the label (class index) from the image file path using class_to_idx mapping
        y = self.class_to_idx[image_filepath.split('/')[-2]]
        
        # Convert the image array to PIL Image and apply the current transformation
        X = Image.fromarray(np.uint8(image)).convert(IMG_MODE)
        X = self.current_transform(X)

        return X, y

def create_data(data_type, data_path,  data_aug, image_shape, b_size):
    """
    Create DataLoader objects for training or testing data.

    Parameters:
        data_type (str): Type of data ('TRAIN_LABEL' or 'TEST_LABEL').
        data_path (str): Path to the directory containing the image data.
        data_aug (bool): Whether to apply data augmentation or not.
        image_shape (tuple): Desired shape of the input images (height, width).
        batch_size (int): Number of samples per batch.

    Returns:
        torch.utils.data.DataLoader: DataLoader object for the specified data type.
    """

    # Get the list of class names from the directory structure
    classes = [image_path.split('/')[-1] for image_path in glob.glob(data_path + '/*')]

    # Get paths of all images
    all_images = [glob.glob(image_path + '/*') for image_path in glob.glob(data_path + '/*')]
    all_images = list(flatten(all_images))

    idx_to_class,class_to_idx = dict(),dict()
    for i, j in enumerate(classes):
        idx_to_class[i] = j
        class_to_idx[j] = i

    # Define image transformations for non-augmented data
    non_aug_tran = transforms.Compose([transforms.Resize((image_shape)),
                                transforms.ToTensor()
                                    ])
    if data_type == TEST_LABEL:
        test_image_paths=all_images
        test_dataset= iNaturalist(test_image_paths,class_to_idx,non_aug_tran)
        test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=True)

        return test_loader

    # Shuffle all image paths to randomly split into training and validation sets
    random.shuffle(all_images)

    tr_paths, v_paths = all_images[:int(TRAIN_DATASET_SPLIT_RATIO*len(all_images))], all_images[int(TRAIN_DATASET_SPLIT_RATIO*len(all_images)):] 

    # Create datasets for training and validation
    tr_data,v_data = iNaturalist(tr_paths,class_to_idx,non_aug_tran),iNaturalist(v_paths,class_to_idx,non_aug_tran)

    if data_aug:
        augu_tran = transforms.Compose([transforms.Resize((image_shape)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                            ])

        tr_data = iNaturalist(tr_paths,class_to_idx,augu_tran)
        v_data = iNaturalist(v_paths,class_to_idx,augu_tran)  
    # Create DataLoader objects for training and validation
    t_loader,v_loader = DataLoader(tr_data, batch_size=b_size, shuffle=True),DataLoader(v_data, batch_size=b_size, shuffle=True)
    return t_loader,v_loader


def inception_v3_model(config_defaults = dict({
        EPOCHS_KEY : 10,
        BATCH_SIZE_KEY: 64,
        LEARNING_RATE_KEY:0.001,
        DATA_AUGMENTATION_KEY: True,
        PRETRAINED_KEY: True,
        FREEZE_KEY:45
    }),isWandb=True):
    """
    Train the neural network model using the specified configurations and hyperparameters.

    Parameters:
    config_defaults (dict): Default parameter contain best parameters for model.
    isWandb (bool): if we don't want to use wandb to log report than pass False

    Returns:
        model
    """

    torch.cuda.empty_cache()
    image_shape = (1,3,299,299)
    test_data_path = TEST_DATA_PATH
    train_data_path = TRAIN_DATA_PATH

    args = DotDict(config_defaults)

    if isWandb:
        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,config = config_defaults)
        args = wandb.config

        wandb.run.name = 'ep-'+str(args.epochs)+'-lr-'+str(args.learning_rate)+'-bs-'+str(args.batch_size) \
             + '-da-'+str(args.data_aug) +'-pretrained-'+str(args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.inception_v3(pretrained=args[PRETRAINED_KEY], progress=True)
    freeze_layers(model, args[FREEZE_KEY])
    model.AuxLogits.fc = nn.Linear(768, 10,bias=True)
    model.fc = nn.Linear(2048, 10, bias=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args[LEARNING_RATE_KEY])

    for epoch in range(args.epochs):
        model.train()
        test_loader = create_data(TEST_LABEL,test_data_path,args[DATA_AUGMENTATION_KEY], image_shape[2:], args[BATCH_SIZE_KEY])
        train_loader, valid_loader = create_data(TRAIN_LABEL,train_data_path,args[DATA_AUGMENTATION_KEY],image_shape[2:], args[BATCH_SIZE_KEY])

        train_correct, train_loss = 0, 0
        total_samples = 0
        for batch_id,(data,label) in enumerate(tqdm(train_loader)):
          
            data = data.to(device=device)
            targets = label.to(device=device)

            loss = nn.CrossEntropyLoss()(scores, targets)
            train_loss += loss.item()
            scores = model(data)
            scores, _ = scores
            scores = F.softmax(scores, dim=1)
            
            _, predictions = scores.max(1)
            train_correct += (predictions == targets).sum()
            total_samples +=  predictions.size(0)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        
        train_loss /= total_samples
        train_acc = round((train_correct / total_samples).item()  * 100, 4)
        
       
        
        val_acc, val_loss = evaluate(device, valid_loader, model)
        test_acc, test_loss = evaluate(device, test_loader, model)
        
        if isWandb:
            wandb.log(
            {TRAIN_ACCURACY_TITLE: train_acc, VALIDATION_ACCURACY_TITLE: val_acc, TEST_ACCURACY_TITLE: test_acc, TRAIN_LOSS_TITLE: train_loss, VALIDATION_LOSS_TITLE: val_loss, TEST_LOSS_TITLE: test_loss}
            )

        print('\nEpoch ', epoch, TRAIN_ACCURACY_TITLE, train_acc, VALIDATION_ACCURACY_TITLE, val_acc, TEST_ACCURACY_TITLE, test_acc, TRAIN_LOSS_TITLE, train_loss, VALIDATION_LOSS_TITLE, val_loss, TEST_LOSS_TITLE, test_loss) 
    return model


'''
if type(args.learning_rate)==type(''):
    args.learning_rate = float(args.learning_rate)
if(type(args.epochs)==type('')):
    args.epochs = float(args.epochs)
if(type(args.batch_size)==type('')):
    args.batch_size = int(args.batch_size)
if(type(args.freeze_k)==type('')):
    args.freeze_k = int(args.freeze_k)

if(type(args.data_aug)==type('')):
    if args.data_aug=='1':
        args.data_aug = True
    else:
        args.data_aug = False

model_configs = dict({
    EPOCHS_KEY : args.epochs,
    BATCH_SIZE_KEY: args.batch_size,
    LEARNING_RATE_KEY:args.learning_rate,
    DATA_AUGMENTATION_KEY: args.data_aug,
    PRETRAINED_KEY: True,
    FREEZE_KEY:args.freeze_k
})
'''

inception_v3_model(isWandb=False)
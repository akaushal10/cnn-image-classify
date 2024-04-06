import cv2
import glob
import random
import numpy as np
import torch
from pandas.core.common import flatten
torch.manual_seed(7)
torch.cuda.empty_cache()
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch import  nn,optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# constants
IMG_MODE = 'RGB'
TRAIN_LABEL = 'train'
TEST_LABEL = 'test'
DATASET_PATH = '/kaggle/input/nature-12k/inaturalist_12K/'
TEST_DATA_PATH = f'{DATASET_PATH}val/'
TRAIN_DATA_PATH = f'{DATASET_PATH}train/'

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

# wandb plot titles
TRAIN_ACCURACY_TITLE = "train_acc"
VALIDATION_ACCURACY_TITLE = "val_acc"
TEST_ACCURACY_TITLE = "test_acc"
TRAIN_LOSS_TITLE = "train_loss"
VALIDATION_LOSS_TITLE = "val_loss"
TEST_LOSS_TITLE = "test_loss"

# Ratio to split train and validation 0.8 means 80% train data and 20% validation data
TRAIN_DATASET_SPLIT_RATIO = 0.8

# Best paramters get by running sweep with different parameters
best_params = dict({
    ACTIVATION_FUNCTION_KEY: RELU_KEY,
    BATCH_NORMALIZATION_KEY: True,
    BATCH_SIZE_KEY: 128,
    DATA_AUGMENTATION_KEY: False,
    DROPOUT_KEY: 0.1,
    EPOCHS_KEY : 30,
    FILTER_ORGANIZATION_KEY: 2,
    LEARNING_RATE_KEY:0.0001,
    DENSE_LAYER_NEURONS_KEY: 512,
    NUMBER_FILTER_KEY: 32,
    SIZE_FILTER_KEY:[11,9,7,5,3]
})

# Default configs for sweep, contain method and params with sweep name
default_sweep_config = {
    "name" : "Assignment2_Part_A_Q2",
    "method" : "bayes",
    'metric': {
        'name': VALIDATION_ACCURACY_TITLE,
        'goal': 'maximize'
    },
    "parameters" : {
        NUMBER_FILTER_KEY: {
            'values': [16, 32, 64, 128]
        },
        ACTIVATION_FUNCTION_KEY: {
            'values': [RELU_KEY, LEAKY_RELU_KEY,GELU_KEY,SILU_KEY,MISH_KEY,ELU_KEY]
        },
        FILTER_ORGANIZATION_KEY: {
            'values': [1, 2, 0.5]
        },
        DATA_AUGMENTATION_KEY: {
            "values": [True,False]
        },
        BATCH_NORMALIZATION_KEY: {
            "values": [True,False]
        },
        DROPOUT_KEY: {
            "values": [0,0.1,0.2,0.3]
        },
        BATCH_SIZE_KEY: {
            "values": [32, 64, 128]
        },
        EPOCHS_KEY : {
            "values" : [10, 15, 20 , 25 , 30]
        },
        LEARNING_RATE_KEY:{
            "values": [0.001,0.0001,0.0003,0.0005]
        },
        SIZE_FILTER_KEY:{
            'values': [[7,5,5,3,3], [11,9,7,5,3]]
        },
        DENSE_LAYER_NEURONS_KEY: {
            "values": [32, 64, 128, 256, 512, 1024]
        }
    }
}

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

def img_prediction_grid(model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape= (1,3,224,224)

    # Get the list of class names from the directory structure
    classes = [image_path.split('/')[-1] for image_path in glob.glob(TEST_DATA_PATH + '/*')]

    #Creating dictionary for class indexes
    idx_to_class = dict()
    for i, j in enumerate(classes):
        idx_to_class[i] = j

    # Loading test data in tensor form
    test_loader = create_data(TEST_LABEL , TEST_DATA_PATH , False , image_shape[2:] , 1) # while loading test data we not require data augmentation hence third argument is False

    # lists used to store images and title of images
    train_img,img_title = [],[]

    model.eval()

    with torch.no_grad():
        for batch_id,(x, y) in enumerate(test_loader):
            x,y = x.to(device=device),y.to(device=device)

            np.array((y.to('cpu')))[0]

            output = model(x)
            _, prediction = output.max(1)

            #converting x into images
            x= torch.squeeze(x)
            im = transforms.ToPILImage()(x).convert(IMG_MODE)

           #setting title of images
            title = str("Actual Label : ") + idx_to_class[np.array((y.to('cpu')))[0]] + "\n" + str("Predicted Label : ") + idx_to_class[np.array((prediction.to('cpu')))[0]]

            #appending images to wandb
            train_img.append(im)
            img_title.append(title)
            if batch_id == 29:
                break
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME)
    wandb.log({"sample image and prediction from the test dataset": [wandb.Image(img, caption=lbl) for img,lbl in zip(train_img,img_title)]})

def run_sweep(sweep_config = default_sweep_config):
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME)
    wandb.agent(sweep_id, train, count = 50)
    wandb.finish()

class DotDict:
    """
    Used to convert dict to an object
    """
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

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

class ConvolutionBlocks(nn.Module):
    """
    A class representing a series of convolutional blocks with optional batch normalization and activation functions.

    Parameters:
        activation (torch.nn.Module): Activation function to be applied after each convolutional layer.
        batch_norm (bool): Flag indicating whether to use batch normalization.
        size_filters (list): List of kernel sizes for each convolutional layer.
        filter_organization (float): Factor by which the number of filters increases in subsequent layers.
        number_filters (int): Number of filters in the first convolutional layer.
        num_conv_layers (int): Number of convolutional layers.

    Attributes:
        activationFn (torch.nn.Module): Activation function to be applied after each convolutional layer.
        num_filters (list): List to store the number of filters in each layer.
        batch_norm (bool): Flag indicating whether to use batch normalization.
        conv1 to conv5 (torch.nn.Conv2d): Convolutional layers.
        pool (torch.nn.MaxPool2d): Max pooling layer.
        batchnorm1 to batchnorm5 (torch.nn.BatchNorm2d): Batch normalization layers.

    Methods:
        forward(x): Forward pass through the convolutional blocks.

    """
    def __init__(self, activation, batch_norm, size_filters, filter_organization, number_filters,num_conv_layers):
        super().__init__()

        # Initialize attributes
        self.activationFn=activation
        self.num_filters=[number_filters]
        self.batch_norm=batch_norm

        # Calculate number of filters for each layer
        for i in range(1,num_conv_layers):
            self.num_filters.append(int(self.num_filters[i-1]*filter_organization))

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.num_filters[0],kernel_size=size_filters[0],stride=(1, 1),padding=(1, 1),bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters[0],out_channels=self.num_filters[1],kernel_size=size_filters[1],stride=(1, 1),padding=(1, 1),bias=False)
        self.conv3 = nn.Conv2d(in_channels=self.num_filters[1],out_channels=self.num_filters[2],kernel_size=size_filters[2],stride=(1, 1),padding=(1, 1),bias=False)
        self.conv4 = nn.Conv2d(in_channels=self.num_filters[2],out_channels=self.num_filters[3],kernel_size=size_filters[3],stride=(1, 1),padding=(1, 1),bias=False)
        self.conv5 = nn.Conv2d(in_channels=self.num_filters[3],out_channels=self.num_filters[4],kernel_size=size_filters[4],stride=(1, 1),padding=(1, 1),bias=False)

        # Define max pooling layer
        self.pool  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Define batch normalization layers if batch_norm is True
        self.batchnorm1 = nn.BatchNorm2d(self.num_filters[0])
        self.batchnorm2 = nn.BatchNorm2d(self.num_filters[1])
        self.batchnorm3 = nn.BatchNorm2d(self.num_filters[2])
        self.batchnorm4 = nn.BatchNorm2d(self.num_filters[3])
        self.batchnorm5 = nn.BatchNorm2d(self.num_filters[4])

    def forward(self, x):
        """
        Perform forward pass through the convolutional blocks.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional blocks.
        """
        if not self.batch_norm: # If batch normalization is not used
            x=self.pool(self.activationFn(self.conv1(x)))
            x=self.pool(self.activationFn(self.conv2(x)))
            x=self.pool(self.activationFn(self.conv3(x)))
            x=self.pool(self.activationFn(self.conv4(x)))
            x=self.pool(self.activationFn(self.conv5(x)))
            return x
        else: # If batch normalization is used
            x= self.pool(self.activationFn(self.batchnorm1(self.conv1(x))))
            x= self.pool(self.activationFn(self.batchnorm2(self.conv2(x))))
            x= self.pool(self.activationFn(self.batchnorm3(self.conv3(x))))
            x= self.pool(self.activationFn(self.batchnorm4(self.conv4(x))))
            x= self.pool(self.activationFn(self.batchnorm5(self.conv5(x))))
            return x

class Model(nn.Module):
    """
    A class representing a convolutional neural network model.

    Parameters:
        image_shape (tuple): Shape of the input images (height, width, channels).
        dropout (float): Dropout probability for regularization.
        activation (str): Name of the activation function to be used.
        batch_norm (bool): Flag indicating whether to use batch normalization in convolutional blocks.
        size_filters (list): List of kernel sizes for each convolutional layer.
        filter_organization (float): Factor by which the number of filters increases in subsequent layers.
        number_filters (int): Number of filters in the first convolutional layer.
        neurons_in_dense_layer (int): Number of neurons in the dense (fully connected) layer.
        num_conv_layers (int): Number of convolutional layers.

    Attributes:
        activation (torch.nn.Module): Activation function to be applied throughout the model.
        conv_blocks (ConvolutionBlocks): Object representing a series of convolutional blocks.
        fully_conn_layer_1 (torch.nn.Linear): Fully connected layer.
        output_layer (torch.nn.Linear): Output layer.
        dropout (torch.nn.Dropout): Dropout layer.
        num_conv_layers (int): Number of convolutional layers.

    Methods:
        forward(x): Forward pass through the model.

    """

    def __init__(self, image_shape,dropout , activation, batch_norm, size_filters, filter_organization,
                  number_filters , neurons_in_dense_layer,num_conv_layers):
        super().__init__()

        # Define activation function based on the provided name
        activationFn = {
            RELU_KEY : nn.ReLU(),
            LEAKY_RELU_KEY : nn.LeakyReLU(),
            GELU_KEY : nn.GELU(),
            SILU_KEY : nn.SiLU(),
            MISH_KEY : nn.Mish(),
            ELU_KEY : nn.ELU()
        }
        self.activation = activationFn[activation]

        # Initialize convolutional blocks
        self.conv_blocks = ConvolutionBlocks(activation = self.activation,
                                             batch_norm= batch_norm,
                                             size_filters= size_filters,
                                             filter_organization= filter_organization,
                                             number_filters= number_filters,
                                             num_conv_layers= num_conv_layers)

        # Calculate the size of the output of convolutional blocks
        sz=self.conv_blocks(torch.zeros(*(image_shape))).data.shape

        # Define fully connected layer
        self.fully_conn_layer_1   = nn.Linear(sz[1] * sz[2] * sz[3],neurons_in_dense_layer,bias=True)

        # Define output layer
        self.output_layer= nn.Linear(neurons_in_dense_layer,10,bias=True)

        # Define dropout layer
        self.dropout=nn.Dropout(p=dropout)
        self.num_conv_layers = num_conv_layers
    def forward(self, x):
        """
        Perform forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.conv_blocks(x)

        # Apply dropout and activation to the fully connected layer
        x = self.dropout(self.activation(self.fully_conn_layer_1(x.reshape(x.shape[0],-1))))

        # Apply softmax activation to the output layer
        x = F.softmax(self.output_layer(x),dim=1)
        return x

def train(config_defaults = best_params,isWandb=True):
    """
    Train the neural network model using the specified configurations and hyperparameters.

    Parameters:
    config_defaults (dict): Default parameter contain best parameters for model.
    isWandb (bool): if we don't want to use wandb to log report than pass False

    Returns:
        model
    """
    torch.cuda.empty_cache()

    # Define image shape and data paths
    image_shape = (1,3,224,224)
    test_data_path = TEST_DATA_PATH
    train_data_path = TRAIN_DATA_PATH

    args = DotDict(config_defaults)

    # Define default configuration parameters
    if isWandb:
        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,config = config_defaults)
        args = wandb.config

        # Set the name of the run
        wandb.run.name = 'ep-'+str(args[EPOCHS_KEY])+'-lr-'+str(args[LEARNING_RATE_KEY])+'-bs-'+str(args[BATCH_SIZE_KEY])+'-act-'+str(args[ACTIVATION_FUNCTION_KEY])+'-drt-'+str(args[DROPOUT_KEY]) \
                        +'-bn-'+ str(args[BATCH_NORMALIZATION_KEY])+ '-da-'+str(args[DATA_AUGMENTATION_KEY])+'-filt_sizes-'+str(args[SIZE_FILTER_KEY]) \
                        + '-filt_org-'+str(args[FILTER_ORGANIZATION_KEY])+'-ini_filt'+str(args[NUMBER_FILTER_KEY])+'-n_d-'+str(args[DENSE_LAYER_NEURONS_KEY])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = Model(image_shape= image_shape,
                  dropout= args[DROPOUT_KEY],
                  activation= args[ACTIVATION_FUNCTION_KEY],
                  batch_norm= args[BATCH_NORMALIZATION_KEY],
                  size_filters= args[SIZE_FILTER_KEY],
                  filter_organization= args[FILTER_ORGANIZATION_KEY],
                  number_filters= args[NUMBER_FILTER_KEY],
                  neurons_in_dense_layer= args[DENSE_LAYER_NEURONS_KEY],
                  num_conv_layers= 5
                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args[LEARNING_RATE_KEY])

    # Iterate over batches in training data
    for epoch in range(args[EPOCHS_KEY]):
        model.train()
        test_loader = create_data(TEST_LABEL,test_data_path,args[DATA_AUGMENTATION_KEY], image_shape[2:], args[BATCH_SIZE_KEY])
        train_loader, valid_loader = create_data(TRAIN_LABEL,train_data_path,args[DATA_AUGMENTATION_KEY],image_shape[2:], args[BATCH_SIZE_KEY])

        train_correct, train_loss = 0, 0
        total_samples = 0
        for batch_id,(data,label) in enumerate(tqdm(train_loader)):

            data = data.to(device=device)
            targets = label.to(device=device)

            scores = model(data)
            loss = nn.CrossEntropyLoss()(scores, targets)
            train_loss += loss.item()

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

# Run With best params
print("Best parameters : ",best_params)
train()

'''
#Question 3 Run sweep
run_sweep(sweep_config = default_sweep_config)
'''

'''
# Question 4-a
# train function default take best parameters
train(isWandb = False)
'''

'''
#Question 4-b
# train function default take best parameters
model = train(isWandb = False)
img_prediction_grid(model)
'''
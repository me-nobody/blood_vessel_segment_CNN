# script for NN classification of Microvascular Proliferation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.functional import normalize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import cv2

from glob import glob

import os
from skimage import io
import time
import copy
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/microvascular_proliferation/logs/NN_classification_mvp_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel (level = 'warning')

# get the the logger with the name 'MVP'
logger = logging.getLogger('MVP')  
# override the logger logging level to INFO
logger.setLevel(logging.INFO)


IMAGE_PATH = "/users/ad394h/Documents/microvascular_proliferation/data/training_images_classification/"

logger.info(f"number of folders in training images {len(os.listdir(IMAGE_PATH))}")

# create sample dataframe with image, path and targets

def create_df(IMAGE_PATH):
  file_list =[]
  directory_list = []
  path_list = []
  for root, _, _ in os.walk(IMAGE_PATH):
    path_list.append(root)
  for item in path_list:
    if item.find("mvp")>1:
      mvp_path = item
      mvp = os.listdir(item)      
      logger.info(f"number of mvp images {len(mvp)}")
    if item.find("other")>1:
      other_path = item
      other = os.listdir(item)
      logger.info(f"number of other images {len(other)}")
  mvp_name = [name[:-4] for name in mvp]
  target_mvp = [1 for i in range(len(mvp))]  
  mvp_path = [mvp_path for i in range(len(mvp))]
  other_name = [name[:-4] for name in other]
  target_other = [0 for i in range(len(other))]
  other_path = [other_path for i in range(len(other))]
  len_idx = len(mvp) + len(other)
  image_name = mvp_name + other_name
  target = target_mvp + target_other
  path = mvp_path + other_path
  
  sample_df = pd.DataFrame({'X': image_name,'path':path,'y':target},index = np.arange(0, len_idx))
  return sample_df

sample_df = create_df(IMAGE_PATH)

# Create training and validation sets and here the target is kept along with the train for the dataset module
X_train, X_test,  = train_test_split(sample_df, test_size=0.25, random_state=42)

# conversion to numpy for smooth loading from the dataloader
X_train = X_train.to_numpy() 
X_test = X_test.to_numpy()


class GBMDataset(Dataset):
  def __init__(self, X,transform=None):
      self.X = X
      self.transform = transform

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    img_x = 224
    img_y = 224
    image = cv2.imread(os.path.join(self.X[idx,1],self.X[idx,0])+".jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(img_x,img_y),cv2.INTER_LINEAR)
    image = torch.from_numpy(image).float()
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))
    img_id = self.X[idx,0]
    target = np.int32(self.X[idx,2])
    target = torch.tensor(target)
    # return {"image": image, "label": target, "img_id": img_id}
    return image,target,img_id
  

# initialize a dataset object
gbm_dataset = GBMDataset(X_train)

BATCH_SIZE = 100
NUM_CLASSES = 2

torch.manual_seed(0)
np.random.seed(0)

train_dataset = GBMDataset(X_train)
test_dataset = GBMDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last= True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")

run_training = False
retrain = False
find_learning_rate = False

# load the model
model = torchvision.models.resnet18(pretrained=True)
# model.load_state_dict(torch.load("/users/ad394h/Documents/microvascular_proliferation/model/base_model/resnet_pretrained.pth"))
num_features = model.fc.in_features # get the input features of the pre-trained model
logger.info(f"model state {model.fc}")


# freeze the weights except the last FC layer
for name, param in model.named_parameters():
  if 'fc' not in name:
    #   print(name, param.requires_grad)
      param.requires_grad=False

# modify the FC layer to have 2 outputs
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),

    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),

    nn.Linear(256, NUM_CLASSES)) # reset the output to the required 2 classes

logger.info(f"model state after updating num_classes {model.fc}")

# initialize weights of the fc layer

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
model = model.to(device)

# check the weights of the modified model
def check_weights(m):
  # Iterate through the children of the sequential module
  for layer in m.children(): 
    # Check if the layer has a weight attribute
    if hasattr(layer, 'weight'):  
      logger.info(f"re-modelled model features")
      logger.info(f"layer {layer}")
      logger.info(f" layer weight data {layer.weight.data.mean(), layer.weight.data.std()}")

check_weights(model)      

# compute the class weights
weights = compute_class_weight(y=sample_df.y.values, class_weight="balanced", classes=sample_df.y.unique())
class_weights = torch.FloatTensor(weights)
if device.type=="cuda":
    class_weights = class_weights.cuda()
logger.info(f"class weights are {class_weights}")     

# create the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# create the scoring metric
def f1_score(preds, targets):
    tp = (preds*targets).sum().to(torch.float32).item()
    fp = ((1-targets)*preds).sum().to(torch.float32).item()
    fn = (targets*(1-preds)).sum().to(torch.float32).item()

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * precision * recall/(precision + recall + epsilon)
    return f1_score

# fit function for the model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    train_f1 = []
    test_f1 = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0
    model.to(device)
    logger.info("model loaded to device")
    fit_time = time.time()
    for e in range(epochs):
      since = time.time()
      running_loss = 0
      train_f1_score = 0
      #training loop
      model.train()
      for i, data in enumerate(train_loader):
          #training phase
          image, target,img_id = data
          image = image.to(device,dtype=torch.float)
          # print(f"image shape while training {image.shape}")
          target = target.to(device,dtype=torch.long)
          #forward
          outputs = model(image)
          # print(f"output shape while training {outputs.shape}")
          print_out = outputs.cpu().detach().numpy()
          # print(f"output of resnet {print_out.shape}")
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, target)
          #evaluation metrics
          train_f1_score += f1_score(preds, target)

          #backward
          loss.backward()
          optimizer.step() #update weight
          optimizer.zero_grad() #reset gradient

          #step the learning rate
          lrs.append(get_lr(optimizer))
          scheduler.step()
          running_loss += loss.item()
      else:
          model.eval()
          test_loss = 0
          test_f1_score = 0
          #validation loop
          with torch.no_grad():
            for i, data in enumerate(test_loader):
              image, target,img_id = data
              image = image.to(device,dtype=torch.float)

              target = target.to(device,dtype=torch.long)
              #forward
              outputs = model(image)

              _, preds = torch.max(outputs, 1)
            #   print(f"image {img_id} is predicted {preds}")
              loss = criterion(outputs, target)
              test_loss += loss.item()
              #evaluation metrics
              test_f1_score += f1_score(preds, target)

      #calculatio mean for each batch
      train_losses.append(running_loss/len(train_loader))
      test_losses.append(test_loss/len(test_loader))

      # calculate the score for each batch
      train_f1.append(train_f1_score/len(train_loader))
      test_f1.append(test_f1_score/len(test_loader))
      logger.info(f"Epoch:{e+1}/{epochs}")
      logger.info(f"Train Loss: {running_loss/len(train_loader)}")
      logger.info(f"Test Loss: {test_loss/len(test_loader)}")
     
      logger.info(f"Train f1:{train_f1_score/len(train_loader)}")
      logger.info(f"Test f1: {test_f1_score/len(test_loader)}")
      logger.info(f"Time: {(time.time()-since)/60}")
      logger.info(f"learning rate:{get_lr(optimizer)}")
      history = {'train_loss' : train_losses, 'test_loss': test_losses,
               'train_f1' :train_f1, 'test_f1':test_f1
               }
    logger.info('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

max_lr = 1e-2
epochs = 200
logger.info(f"recorded epochs in beginning are {epochs}")
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                            steps_per_epoch=len(train_loader))

filename = "history_unet_effi_2class_1024px_getOutput_" + str(timestr) + ".csv"

history = fit(epochs, model, train_loader, test_loader, criterion, optimizer, sched)

timestr = time.strftime("%Y%m%d-%H%M%S")

OUTPUT_PATH = '/users/ad394h/Documents/microvascular_proliferation/logs/'


try:    
    # pass
    torch.save(model, '/users/ad394h/Documents/microvascular_proliferation/model/nn_classification_mvp_{}.pt'.format(timestr))
except Exception as e:
    logger.info(f"model couldn't be saved due to {e}")    
try:
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(OUTPUT_PATH,filename),index=False)
except Exception as e:
    logger.info(f"{e} in charting history")


def plot_loss(history):
    plt.plot(history['test_loss'], label='test', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_f1'], label='Train F1', marker='*')
    plt.plot(history['test_f1'], label='Test F1',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()



plot_loss(history)
plt.savefig("/users/ad394h/Documents/microvascular_proliferation/results/nn_mvp_classification_loss_{}.jpg".format(timestr))
plt.clf() # clear the above plot
plot_score(history)
plt.savefig("/users/ad394h/Documents/microvascular_proliferation/results/nn_mvp_classification_score_{}.jpg".format(timestr))

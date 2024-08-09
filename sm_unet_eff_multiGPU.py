import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image


import os
import logging
import hostlist # separately pip installed

import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import normalize

import torch.optim as optim
# packages for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



import segmentation_models_pytorch as smp

import cv2

from sklearn.model_selection import train_test_split


timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/segment_blood_vessels/logs/sm_unet_effi_3_class_{}.txt'.format(timestr)

model_name = "unet_multigpu"

logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


plt.set_loglevel (level = 'warning')

"""Segmentation Models PyTorch UNET Efficientnet b7 with 3 output classes. This model is trained on 
   augmented data and multiple GPUs. Custom conda environment /mnt/scratch/users/ad394h/sharedscratch/anu"""



# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,
#     activation = None,               # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     encoder_depth =5,
#     decoder_channels = [256,128,64,32,16],
#     classes=3,                      # model output channels (number of classes in your dataset)
# )


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")


# access google drive
IMAGE_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/images/'
logger.info(f' the image file numbers are {len(os.listdir(IMAGE_PATH))}')

MASK_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/masks/'
logger.info(f'the mask file numbers are {len(os.listdir(MASK_PATH))}')

OUTPUT_PATH = '/users/ad394h/Documents/segment_blood_vessels/logs/'

base_model_path = '/users/ad394h/Documents/segment_blood_vessels/models/models_2_test/Unet_efficientnet_b7_3_classes.pt'
logger.info(f"base model path {base_model_path}")
model = torch.load(base_model_path)

# "http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html"
# code to implement distributed computation. below is the code to recover values of the environment for later use

# get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
# world_size = int(os.environ['WORLD_SIZE']) # key doesn't exist
 
# get node list from slurm
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
 
# get IDs of reserved GPU
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
 
# define MASTER_ADD & MASTER_PORT
os.environ['MASTER_ADDR'] = hostnames[0]
os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node

# log all the environment variables

logger.info(f"SLURM rank is {rank}")
logger.info(f"SLURM local rank is {local_rank}")
logger.info(f"SLURM tasks are {size}")
logger.info(f"SLURM cpus per task {cpus_per_task}")

# Initialise the process group (i.e. the number of processes, the protocol of collective communications 
# or backend, …). The backends possible are NCCL, GLOO and MPI. NCCL is recommended both for the 
# performance and the guarantee of correct functioning 

dist.init_process_group(backend = 'nccl',
                        init_method = 'env://',
                        world_size = size,
                        rank = rank)

# set the device to cuda

torch.cuda.set_device(local_rank)
gpu = torch.device("cuda")
model = model.to(gpu)


# Transform the model into distributed model associated with a GPU

check_point = '/users/ad394h/Documents/segment_blood_vessels/models/ddp_models.checkpoint'

try:
    ddp_model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)
    logger.info("distributed model created")
    torch.save(ddp_model.state_dict(), check_point)
except Exception as e:
    logger.info("distributed model not created due to {e}")


# We have to now frame the dataset class for our images

n_classes = 3

def create_df(image_path,mask_path):
  image_name = []
  mask_name = []

  for dirname, _, img_names in os.walk(image_path):
    img_names= img_names
  img_names.sort()

  for dirname, _, msk_names in os.walk(mask_path):
    msk_names= msk_names
  msk_names.sort()


  for img_name in img_names:
    if img_name in msk_names:
      msk_idx = msk_names.index(img_name)
      msk_name = msk_names[msk_idx]
      image_name.append(img_name)
      mask_name.append(msk_name)

  logger.info(len(image_name))
  logger.info(len(mask_name))

  return pd.DataFrame({'X': image_name,'y':mask_name},index = np.arange(0, len(image_name)))

df = create_df(IMAGE_PATH,MASK_PATH)

logger.info(f"dataframe shape is {df.shape}")

# split data
X_train, X_test, y_train, y_test = train_test_split(df['X'].values, df['y'].values, test_size=0.25, random_state=19)

logger.info(f'Train Size   : {len(X_train)}')
logger.info(f'Test Size    : {len(X_test)}')


# DATASET 

class SlideDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_x = 224
        img_y = 224
        img = cv2.imread(self.img_path + self.X[idx]) # reads as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(img_x,img_y),cv2.INTER_LINEAR)
        #img = Image.fromarray(img)
        
        mask = cv2.imread(self.mask_path + self.X[idx])
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(img_x,img_y),cv2.INTER_LINEAR)
        x = mask.shape[0]
        y = mask.shape[1]
        for i in range(x):
            for j in range(y):
                if mask[i,j] >= 3 and mask[i,j] <=70: # already the mask is binarized during augmentation
                    mask[i,j] = 0
                elif mask[i,j] > 70 and mask[i,j] <= 184:
                    mask[i,j] = 1
                elif mask[i,j] > 184:
                    mask[i,j] = 2

        
        img = torch.from_numpy(img).float()
        img = normalize(img,dim=0)
        img = torch.permute(img,(2,0,1))
        mask = torch.from_numpy(mask).long()
        return img, mask


# datasets
train_set = SlideDataset(IMAGE_PATH, MASK_PATH, X_train)
test_set = SlideDataset(IMAGE_PATH, MASK_PATH, X_test)

slide_image, slide_mask = train_set.__getitem__(10)

logger.info(f"slide image shape {slide_image.shape}")
logger.info(f"slide mask shape {slide_mask.shape}")

# for distributed processing we have to modify the dataloader

# dataloader

batch_size = 50
batch_size_per_gpu = batch_size // size # each GPU will get 50 images
 
 
# Data loading code

train_sampler = DistributedSampler(train_set,
                                   num_replicas = size,
                                   rank=rank,
                                   shuffle=True)
 
train_loader = DataLoader(dataset=train_set,
                         batch_size=batch_size_per_gpu,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True,
                         sampler=train_sampler)

test_sampler = DistributedSampler(test_set,
                                   num_replicas = size,
                                   rank=rank,
                                   shuffle=True)
 
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size_per_gpu,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True,
                         sampler=test_sampler)




# scoring metric

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=3):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

# Training

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epoch, ddp_model, train_loader, test_loader,criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    train_ious = []
    test_ious = []

    epoch_l = []

    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve = 0

    
    if check_point is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        ddp_model.load_state_dict(torch.load(check_point, map_location=map_location))
        map_locations = [item for item in map_location.values()]
        for location in map_location:
            logger.info(f"distributed model loaded at {location}")
    
    # training (timers and display handled by process 0)
    if rank == 0:
        fit_time = time.time()  
        logger.info(f"starting training for rank {rank}")
    elif rank != 0:
        fit_time = time.time()
        logger.info(f"starting training for rank {rank}")    
    else:
        fit_time = 0
        logger.info(f"fit time set to 0")    
        total_step = len(train_loader)


    for e in range(epoch):      

        start_epoch = time.time()
        logger.info(f"starting epoch for rank {rank}")        

        # extra step for DDP
        train_sampler.set_epoch(epoch) # necessary for shuffing across GPUs

        running_loss = 0
        train_accuracy = 0
        iou_score = 0

        # training loop
        ddp_model.train()
        dist.barrier() # makes sure that only the 1st process processes the data
        for i, (images,masks) in enumerate(train_loader):
            
            #training phase           
            image = images.to(gpu,non_blocking=True)
            mask = masks.to(gpu,non_blocking=True)
            
            #forward
            output = ddp_model(image)            
            loss = criterion(output, mask)

            #evaluation metrics                
            train_accuracy += pixel_accuracy(output, mask)
            iou_score += mIoU(output, mask)
            
            #backward
            loss.backward()  # compute gradient of the loss w.r.t. to the parameters
            optimizer.step() # update weight
            optimizer.zero_grad() # reset gradient

            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
            running_loss += loss.item()            
            

        # validation loop 
        else:
            
            test_loss = 0
            test_accuracy = 0   
            test_iou_score = 0

            ddp_model.eval()
            dist.barrier() # makes sure that all processes are synchronized
            with torch.no_grad():
                for i, (images,masks) in enumerate(test_loader):
                    # reshape to 9 patches from single image, delete batch size
                    
                    image = images.to(gpu,non_blocking=True)
                    mask = masks.to(gpu,non_blocking=True)

                    # forward
                    output = ddp_model(image)
                    #output = torch.argmax(output,dim = 1)
                    loss = criterion(output, mask)
                    test_loss += loss.item()
                    
                    #evaluation metrics                
                    test_accuracy += pixel_accuracy(output, mask)
                    test_iou_score += mIoU(output, mask)
                
            dist.barrier()               
    
            test_loss = torch.Tensor([ loss ]).cuda()    # collects all scalar from all GPUs as tensor array     
            dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)  # mean of the tensor array      
            test_loss = test_loss.item() # convert tensor to a float

            test_accuracy = torch.Tensor([test_accuracy]).cuda()
            dist.all_reduce(test_accuracy,op=dist.ReduceOp.SUM)
            test_accuracy = test_accuracy.item()

            test_iou_score = torch.Tensor([test_iou_score]).cuda()
            dist.all_reduce(test_iou_score,op=dist.ReduceOp.SUM)
            test_iou_score = test_iou_score.item()


            if rank == 0:
                logger.info(f"length of train loader {len(train_loader)}")
                logger.info(f"length of test loader {len(test_loader)}")

                running_loss = running_loss/size
                running_loss = running_loss/len(train_loader)

                test_loss = test_loss/size     
                test_loss = test_loss/len(test_loader)          
                
                
                train_accuracy = train_accuracy/len(train_loader)
                train_accuracy = train_accuracy/size
                
                test_accuracy = test_accuracy/len(test_loader)
                test_accuracy = test_accuracy/size
                

                test_iou_score = test_iou_score/len(test_loader)
                test_iou_score = test_iou_score/size

                logger.info(f"Epoch:{e+1}/{epoch}")
                logger.info(f"Train Loss : {running_loss:,.2f}")  
                logger.info(f"Test Loss : {test_loss:,.2f}")

                logger.info(f"Train accuracy : {train_accuracy:,.2f}")
                logger.info(f"Test accuracy : {test_accuracy:,.2f}")

                logger.info(f"Train IoU score : {iou_score:,.2f}")
                logger.info(f"Test IoU score : {test_iou_score:,.2f}")

                
                train_losses.append(running_loss)     
                test_losses.append(test_loss)
                
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)


                train_ious.append(iou_score)
                test_ious.append(test_iou_score)

                epoch_l.append(int(e)+1)

                stop_epoch = time.time()
                logger.info(f"stopping epoch for rank {rank}")
                try:
                    torch.save(ddp_model.state_dict(), '/users/ad394h/Documents/segment_blood_vessels/models/sm_unet_eff_b7_3 cl_50_bs_augdata_{}.checkpoint'.format(model_name))
                except Exception as e:
                    logger.info(f"model couldn't be saved due to {e}")
                if stop_epoch > 0:  
                    logger.info(f"Time : {(stop_epoch -start_epoch)//60} min")                    
                else:
                    logger.info(f"stop epoch not calculated {stop_epoch}")    
                    logger.info(f"learning rate:{get_lr(optimizer)}")

                logger.info('Total time: {:.2f} m' .format((stop_epoch- fit_time)//60))

    history = { 'epochs' :epoch_l,
                'train_loss' : train_losses,'test_loss' : test_losses,
                'train_accuracy' :train_accuracies, 'test_accuracy':test_accuracies,
                'train_ious' :train_ious, 'test_ious' : test_ious
                }
    
    return history    

        # if rank == 0:
        #     stop_epoch = time.time()
        #     logger.info(f"stopping epoch for rank {rank}")
        #     try:
        #         torch.save(ddp_model.state_dict(), '/users/ad394h/Documents/segment_blood_vessels/models/sm_unet_eff_b7_3 cl_50_bs_augdata_{}.checkpoint'.format(model_name))
        #     except Exception as e:
        #         logger.info(f"model couldn't be saved due to {e}") 
        # elif rank != 0:
        #     stop_epoch = time.time()    
        #     logger.info(f"stopping epoch for rank {rank}")
        # else:
        #     stop_epoch = 0    
        #     logger.info(f"stop epoch set to 0")   
    

def plot_loss(history):    
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.plot( history['test_loss'], label='test', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history['train_accuracy'], label='train_accuracy', marker='*')
    plt.plot(history['test_accuracy'], label='test_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_ious'], label='train_mIoU', marker='*')
    plt.plot(history['test_ious'], label='test_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

if __name__ == '__main__':
    # get distributed configuration from Slurm environment
    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    # display info
    if rank == 0:
        logger.info(f"Training on {len(hostnames)} nodes and {size} processes, master node is {MASTER_ADDR}")
        logger.info(f"Slurm node ID is {NODE_ID}")
        logger.info("- Process {} corresponds to GPU {} of node {}".format(rank, local_rank, NODE_ID))


    # set the hyperparameters
    max_lr = 1e-3
    epoch = 5
    weight_decay = 1e-4

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    filename = "history_unet_eff_b7_3_cl_50_bs_augdata" + str(timestr) + ".csv"

    history = fit(epoch, ddp_model, train_loader, test_loader,criterion, optimizer, sched)
    
      
    try:
        history = pd.DataFrame(history)
        history.to_csv(os.path.join(OUTPUT_PATH,filename),index=False)
    except Exception as e:
        logger.info(f"{e} in charting history")

    plot_loss(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_loss_aug_data{}.jpg".format(model_name))
    plt.clf() # clear the old plot
    plot_score(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_score_batch_{}.jpg".format(model_name))
    plt.clf()
    plot_acc(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_accuracy_batch_{}.jpg".format(model_name))

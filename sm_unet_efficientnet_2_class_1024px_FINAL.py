import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/segment_blood_vessels/logs/sm_unet_effi_2_class_1024px_final{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel (level = 'warning')

# get the the logger with the name 'PIL'
pil_logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)



# 2 class Unet model

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import normalize

import cv2

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")


# access google drive
IMAGE_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/1024px_to_256px/'
logger.info(f' the image file numbers are {len(os.listdir(IMAGE_PATH))}')

MASK_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/1024px_mask_to_256px_mask/'
logger.info(f'the mask file numbers are {len(os.listdir(MASK_PATH))}')

OUTPUT_PATH = '/users/ad394h/Documents/segment_blood_vessels/logs/'

model = torch.load('/users/ad394h/Documents/segment_blood_vessels/models/models_2_test/Unet_efficientnet_2_class_base_model.pt')

# model = smp.Unet(
#     encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",            # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,
#     activation = None,                     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     encoder_depth =5,
#     decoder_channels = [256,128,64,32,16],
#     classes=2,                             # model output channels (number of classes in your dataset)
# )


# We have to now frame the dataset class for our images

n_classes = 2

# this part of the code synchronizes the names in the image and mask. the names have to be same
# in the 2 folders

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

#split data
X_train, X_test, y_train, y_test = train_test_split(df['X'].values, df['y'].values, test_size=0.25, random_state=19)

logger.info(f'Train Size   : {len(X_train)}')
logger.info(f'Test Size    : {len(X_test)}')


"""### DATASET"""

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
        img = cv2.imread(self.img_path + self.X[idx])
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
                if mask[i,j] <=200: 
                    mask[i,j] = 0
                else:
                    mask[i,j] = 1   
        
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

# dataloader
batch_size= 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#dummy_img, dummy_mask = next(iter(train_loader))

#logger.info(f"loader image shape {dummy_img.shape}")
#logger.info(f"loader mask shape {dummy_mask.shape}")

""" call the model without training"""

def predict_image_mask(model, image, mask):
    img_x = image.shape[0]
    img_y = image.shape[1]
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask,(img_x,img_y),cv2.INTER_LINEAR)
    model.eval()
    image = torch.from_numpy(image).float()
    mask = torch.from_numpy(mask).long()
    #print(f"image shape is {img.shape}")
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))

    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        logger.info(f"output shape {output.shape}")
        masked = torch.argmax(output, dim=1)
        logger.info(f"masked shape {masked.shape}")
        masked = masked.cpu().squeeze(0)
    return mask,output,masked

# mask,output,predicted_mask = predict_image_mask(model,image,mask)


# Training

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    test_iou = []; test_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    #logger.info("model loaded to device")
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        for i, data in enumerate(train_loader):
            # logger.info(f'sample {i}')
            #training phase
            image_tiles, mask_tiles = data
            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            #forward
            output = model(image)
            #logger.info(f'output of base model {output.shape}')
            loss = criterion(output, mask)
            #evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
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
            test_accuracy = 0
            test_iou_score = 0
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    #output = torch.argmax(output,dim = 1)
                    loss = criterion(output, mask)
                    #evaluation metrics
                    test_iou_score +=  mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))


            if min_loss > (test_loss/len(test_loader)):
                logger.info('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(test_loader))))
                min_loss = (test_loss/len(test_loader))
                decrease += 1
                if decrease % 20 == 0:
                    logger.info('saving model...')
                    torch.save(model, '/users/ad394h/Documents/segment_blood_vessels/models/Unet-efficienet_b7_mIoU-{:.3f}.pt'.format(test_iou_score/len(test_loader)))


            if (test_loss/len(test_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(test_loader))
                logger.info(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 20:
                    logger.info(f'Loss not decrease for 20 times, Stop Training')
                    

            #iou
            test_iou.append(test_iou_score/len(test_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            test_acc.append(test_accuracy/ len(test_loader))
            logger.info(f"Epoch:{e+1}/{epochs}")
            logger.info(f"Train Loss: {running_loss/len(train_loader)}")
            logger.info(f"Test Loss: {test_loss/len(test_loader)}")
            logger.info(f"Train mIoU:{iou_score/len(train_loader)}")
            logger.info(f"Test mIoU: {test_iou_score/len(test_loader)}")
            logger.info(f"Train Acc:{accuracy/len(train_loader)}")
            logger.info(f"Test Acc:{test_accuracy/len(test_loader)}")
            logger.info(f"Time: {(time.time()-since)/60}")
            logger.info(f"learning rate:{get_lr(optimizer)}")
    history = {'train_loss' : train_losses, 'test_loss': test_losses,
               'train_miou' :train_iou, 'test_miou':test_iou,
               'train_acc' :train_acc, 'test_acc':test_acc
               }
    logger.info('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

max_lr = 1e-2
epoch = 100
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

filename = "history_unet_effi_2class_1024px_final_" + str(timestr) + ".csv"

history = fit(epoch, model, train_loader, test_loader, criterion, optimizer, sched)

timestr = time.strftime("%Y%m%d-%H%M%S")
try:
    torch.save(model, '/users/ad394h/Documents/segment_blood_vessels/models/sm_unet_eff_b7_2class_1024px_final_{}.pt'.format(timestr))
except Exception as e:
    logger.info(f"model couldn't be saved due to {e}")    
try:
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(OUTPUT_PATH,filename),index=False)
except Exception as e:
    logger.info(f"{e} in charting history")



#model = torch.load('Unet-Resnet-18_may_24.pt')

def plot_loss(history):
    plt.plot(history['test_loss'], label='test', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['test_miou'], label='test_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['test_acc'], label='test_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

plot_loss(history)
plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_2class_1024px_loss_final_{}.jpg".format(timestr))
plt.clf() # clear the above plot
plot_score(history)
plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_2class_1024px_score_final_{}.jpg".format(timestr))
plt.clf()
plot_acc(history)
plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_2class_1024px_accuracy_final_{}.jpg".format(timestr))

def predict_image_mask(model, image, mask):
    img_x = image.shape[0]
    img_y = image.shape[1]
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask,(img_x,img_y),cv2.INTER_LINEAR)
    model.eval()
    image = torch.from_numpy(image).float()
    mask = torch.from_numpy(mask).long()
    #logger.info(f"image shape is {img.shape}")
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))

    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        logger.info(f"output shape {output.shape}")
        masked = torch.argmax(output, dim=1)
        logger.info(f"masked shape {masked.shape}")
        masked = masked.cpu().squeeze(0)
    return mask,output,masked

#mask,output,predicted_mask = predict_image_mask(model,image,mask)

def calc_resize(image):
  img_x = image.shape[0]//32
  img_y = image.shape[1]//32
  rem_x = img_x%32
  rem_y = img_y%32
  if rem_x !=0:
    logger.info(f"remainder {rem_x}")
    logger.info(f"resize X to {img_x*32}")
  else:
    logger.info("X divides by 32")
  if rem_y !=0:
    logger.info(f"remainder {rem_y}")
    logger.info(f"resize Y to {img_y*32}")
  else:
    logger.info("Y divides by 32")


def predict_mouse_bv(model, image):
    model.eval()
    image = torch.from_numpy(image).float()
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))

    model.to(device); image=image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        logger.info(f"output shape {output.shape}")
        masked = torch.argmax(output, dim=1)
        logger.info(f"masked shape {masked.shape}")
        masked = masked.cpu().squeeze(0)
    return masked


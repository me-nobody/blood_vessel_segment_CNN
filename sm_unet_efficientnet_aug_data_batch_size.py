import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/segment_blood_vessels/logs/unet_batchsize_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



"""THIS MODEL HAS THE IMPROVED THRESHOLDS"""

import segmentation_models_pytorch as smp

# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,
#     activation = None,               # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     encoder_depth =5,
#     decoder_channels = [256,128,64,32,16],
#     classes=3,                      # model output channels (number of classes in your dataset)
# )



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
IMAGE_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/augmented_images/'
logger.info(f' the image file numbers are {len(os.listdir(IMAGE_PATH))}')

MASK_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/augmented_masks/'
logger.info(f'the mask file numbers are {len(os.listdir(MASK_PATH))}')

OUTPUT_PATH = '/users/ad394h/Documents/segment_blood_vessels/logs/'

# We have to now frame the dataset class for our images

n_classes = 3

def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename[:-4])

    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df()

logger.info(f"dataframe shape is {df.shape}")

#split data
X_train, X_test = train_test_split(df['id'].values, test_size=0.25, random_state=19)

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
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(img_x,img_y),cv2.INTER_LINEAR)
        #img = Image.fromarray(img)
        
        mask = cv2.imread(self.mask_path + self.X[idx] + '.jpg')
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


model = torch.load('/users/ad394h/Documents/segment_blood_vessels/models/models_2_test/Unet_efficientnet_b7_noweights.pt')


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
    epoch_l = []
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
            loss.backward() # Compute gradient of the loss w.r.t. to the parameters  
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
                    logger.info('was saving model...')
                    torch.save(model, '/users/ad394h/Documents/segment_blood_vessels/models/Unet-efficientnet_v2_mIoU-{:.3f}.pt'.format(test_iou_score/len(test_loader)))


            if (test_loss/len(test_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(test_loader))
                logger.info(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 20:
                    logger.info(f'Loss not decrease for 7 times, Stop Training')
                    

            #iou
            test_iou.append(test_iou_score/len(test_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            test_acc.append(test_accuracy/ len(test_loader))
            epoch_l.append(int(e)+1)

        logger.info(f"Epoch:{e+1}/{epochs}")
        logger.info(f"Train Loss: {running_loss/len(train_loader)}")
        logger.info(f"Test Loss: {test_loss/len(test_loader)}")
        logger.info(f"Train mIoU:{iou_score/len(train_loader)}")
        logger.info(f"Test mIoU: {test_iou_score/len(test_loader)}")
        logger.info(f"Train Acc:{accuracy/len(train_loader)}")
        logger.info(f"Test Acc:{test_accuracy/len(test_loader)}")
        logger.info(f"Time: {(time.time()-since)/60}")

    history = {'epochs' :epoch_l,
               'train_loss' : train_losses, 'test_loss': test_losses,
               'train_miou' :train_iou, 'test_miou':test_iou,
               'train_acc' :train_acc, 'test_acc':test_acc
               }
    logger.info('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history


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


# dataloader
batch_size_list = [5,10,20,50]



for batch_size in batch_size_list:
    logger.info(f"the batch size is {batch_size}")
    file_name = "history_batchsize_" + str(batch_size) + ".csv"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    max_lr = 1e-3
    epoch = 50
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))
    try:
        history = fit(epoch, model, train_loader, test_loader, criterion, optimizer, sched)
        history = pd.DataFrame(history)
        history.to_csv(os.path.join(OUTPUT_PATH,file_name),index=False)
    except Exception as e:
        logger.info(f"{e} in charting history")

    try:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        torch.save(model, '/users/ad394h/Documents/segment_blood_vessels/models/sm_unet_efficientnet_{}.pt'.format("batch_size_"+str(batch_size)))
    except Exception as e:
        logger.info(f"{e} in saving model")

    plot_loss(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_loss_batch_{}.jpg".format("batch_size_"+str(batch_size)))
    plt.clf() # clear the old plot
    plot_score(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_score_batch_{}.jpg".format("batch_size_"+str(batch_size)))
    plt.clf()
    plot_acc(history)
    plt.savefig("/users/ad394h/Documents/segment_blood_vessels/tests/unet_efficientnet_accuracy_batch_{}.jpg".format("batch_size_"+str(batch_size)))











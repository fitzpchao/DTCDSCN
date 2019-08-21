import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_sexp6 import *
from model_sexp6 import CDNet34
from utils_sexp6 import Trainer
from cdloss import cdloss
BATCH_SIZE=16
TRAIN_LIST="/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/csv/train_4.csv"
VAL_LIST="/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/csv/val_4.csv"
ROOT_BE='/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/beforechange'
ROOT_AF='/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/afterchange'
ROOT_MASK='/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/label_change_new'
#SAVE_PATH='E:/pc/cd_pixel/checkpoints_paper/paperexp5'
ROOT_MASK_2011='/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/label2011_new'
ROOT_MASK_2016='/home/zhan/pchao/files/mycode/changedetectiondata/Building change detection dataset256new/label2016_new'
SAVE_PATH='/home/zhan/pchao/files/mycode/cd_pixel/checkpoints/sexp6_4'
def get_dataloader(batch_size):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageFolder(TRAIN_LIST,ROOT_BE,ROOT_AF,ROOT_MASK,ROOT_MASK_2011,ROOT_MASK_2016
                      ),
        batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder_val(VAL_LIST,ROOT_BE,ROOT_AF,ROOT_MASK,ROOT_MASK_2011,ROOT_MASK_2016
                      ),
        batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader
def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)
    #model= DinkNet34(num_classes=1)
    model=CDNet34(in_channels=3)
    #optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9,weight_decay=1e-4)
    optimizer=optim.Adam(params=model.parameters(),lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [350], 0.1)
    trainer = Trainer(model, optimizer,cdloss ,save_freq=10,save_dir=SAVE_PATH)
    trainer.loop(400, train_loader, test_loader,scheduler=scheduler)


if __name__ == '__main__':
    main(BATCH_SIZE)
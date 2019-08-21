import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import time
import torch.distributed as dist
from losses import TripletMarginLoss
from distributed_utils import average_gradients

class Trainer(object):
    def __init__(self, model, optimizer, loss_f, logger,args,rank):
        self.model = model
        #self.model.load_state_dict(torch.load("E:\pc\cd_pixel\checkpoints\exp8_2\ model_43.pkl")['weight'])
        self.optimizer = optimizer
        self.loss_f = TripletMarginLoss(0.5).cuda()
        self.save_dir = args.save_path
        self.save_freq = args.save_step
        if(rank==0):
            self.writer = SummaryWriter()
        self.epochs=args.epochs
        self.logger=logger
        self.rank=rank
        self.args=args

    def select_triplet(self,x, target):
        # x:[B,C,H,W],traget:[B,1,H,W]
        x = x.detach().cpu().data.numpy()
        target = F.adaptive_max_pool2d(target,(33,33))
        target = target.detach().cpu().data.numpy()
        B, C, H, W = x.shape

        tripletlist_anchor=[]
        tripletlist_p=[]
        tripletlist_n=[]
        for i in range(B):

            anchor = x[i]
            label = target[i]

            # print label.shape
            anchor_re = anchor.reshape([C, H * W])
            Y = label.reshape([H * W])
            if(len(Y[Y==1])<2):
                continue
            if(len(Y[Y==0])<2):
                continue
            [dim, sampleNum] = anchor_re.shape

            for k in range(0, sampleNum):
                HitDist = float('inf') * np.ones(sampleNum, dtype=float)
                MissDist = float('inf') * np.ones(sampleNum, dtype=float)
                ak = anchor_re[:, k]
                akd = anchor_re - ak[:, np.newaxis] * np.ones([1, sampleNum], dtype=float)
                akd = akd * akd
                Distik = np.sum(akd, 0)
                SameLabel = [h for (h, w) in enumerate(Y) if w == Y[k]]
                DiffLabel = [h for (h, w) in enumerate(Y) if w != Y[k]]
                a = len(SameLabel)
                HitDist[SameLabel] = Distik[SameLabel]
                MissDist[DiffLabel] = Distik[DiffLabel]
                HitDist[k] = float('inf')
                SortedHitIndex = np.argsort(HitDist)
                SortedMissIndex = np.argsort(MissDist)
                HitSet = SortedHitIndex[a - 2]
                MissSet = SortedMissIndex[0]

                tripletlist_anchor.append( i*k + k)
                tripletlist_p.append(i*k + HitSet)
                tripletlist_n.append( i*k + MissSet)

        return tripletlist_anchor,tripletlist_p,tripletlist_n

    def _iteration(self, data_loader, ep ,is_train=True):
        loop_loss = []
        matrixs = np.zeros([2,2],np.float32)

        for i,(img1,img2,target) in enumerate(data_loader):

            img1,img2,target = img1.cuda(),img2.cuda(),target.cuda()
            target1 = target.cpu().data.numpy()
            self.logger.info(len(target1[target1 > 0]))
            output1,output2= self.model(img1,img2)
            out=torch.cat([output1,output2],1)
            index_a,index_p,index_n=self.select_triplet(out,target)
            self.logger.info(len(index_a))
            B,C,H,W = out.size()
            out = out.permute([0, 2, 3, 1])
            out = out.contiguous().view(B*H*W,-1)
            if(index_a==[]):
                loss = 0
                loss_step=0
            else:
                anchor=out[index_a]
                positive=out[index_p]
                negative = out[index_n]
                loss = self.loss_f(anchor,positive,negative)
                loss_step = loss.data.item()
            if(self.rank==0 and is_train):
                self.logger.info(">>>[{}/{}]Train-loss:{}".format(i,len(data_loader),loss_step))
            if(self.rank==0 and not is_train):
                self.logger.info(">>>[{}/{}]Test-loss:{}".format(i,len(data_loader),loss_step))

            loop_loss.append(loss_step / len(data_loader))
            if is_train:
                if not isinstance(loss,int):
                    self.optimizer.zero_grad()
                    loss.backward()
                    average_gradients(self.model)
                    self.optimizer.step()

        if is_train:
            if(self.rank==0):
                self.writer.add_scalar('train/loss_epoch', sum(loop_loss), ep)
        else:
            if (self.rank == 0):
                self.writer.add_scalar('test/loss_epoch', sum(loop_loss), ep)
        if (self.rank == 0):
            mode = "train" if is_train else "test"
            self.logger.info(">>>[{mode}] loss: {loss}".format(mode=mode,loss=sum(loop_loss)))
        return loop_loss

    def train(self, data_loader,ep):
        self.model.train()
        self.model = self.model.cuda()
        with torch.enable_grad():
            loss = self._iteration(data_loader,ep)
            #pass

    def test(self, data_loader,ep):
        self.model.eval()
        self.model = self.model.cuda()
        with torch.no_grad():
            loss = self._iteration(data_loader,ep,is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None):
        train_data.sampler.set_epoch(self.epochs)
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            if (self.rank == 0):
                self.logger.info("epochs: {}".format(ep))
            if (ep % self.args.print_freq == 0):
                start_time = time.time()
                self.test(test_data, ep)
                end_time = time.time()
                if (self.rank == 0):
                    self.logger.info("epochs: {}".format(ep))
                    self.logger.info('total_time:{}s'.format(end_time - start_time))

            start_time=time.time()
            self.train(train_data,ep)
            end_time=time.time()
            if(self.rank==0):
                self.logger.info("epochs: {}".format(ep))
                self.logger.info('total_time:{}s'.format(end_time-start_time))
            if (ep % self.save_freq == 0 and self.rank ==0):

                self.save(ep)


    def save(self, epoch, **kwargs):
        model_out_path = self.save_dir
        state = {"epoch": epoch, "weight": self.model.cpu().state_dict()}
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        torch.save(state, model_out_path + '/model_{epoch}.pkl'.format(epoch=epoch))
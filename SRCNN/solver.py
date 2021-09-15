from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from progress_bar import progress_bar

from Trainer import Trainer #==> add

class SRCNNTrainer(Trainer):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
        self.config = config
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = Net(num_channels=1, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.set_optimizer() #==> Add

    def train(self):
        self.model.train()
        train_loss = 0
        avg_psnr   = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            total_time = progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr

        avg_psnr = avg_psnr / len(self.training_loader)
        avg_loss = train_loss / len(self.training_loader)
        return [avg_loss, avg_psnr, total_time]

    def test(self):
        self.model.eval()
        avg_psnr = 0
        test_loss = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                test_loss += mse.item()
                total_time = progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

        avg_psnr = avg_psnr / len(self.testing_loader)
        avg_loss = test_loss / len(self.testing_loader)
        return [avg_loss, avg_psnr, total_time]

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            train_loss = self.train()
            test_loss = self.test()
            self.scheduler.step(epoch)
            self.save_model(epoch=epoch, error_loss=train_loss, test_loss=test_loss)

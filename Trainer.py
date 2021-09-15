from __future__ import print_function

import collections
import torch
import glob
import errno

import os,sys,inspect
current_dir  = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir   = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from optimizerlib.lamb.lamb import Lamb, log_lamb_rs

CHECKPOINT_DIR = 'checkpoints'
class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.config = None
        self.CUDA   = False
        self.model  = None
        self.device = 'cpu'
        self.lr     = 0.01
        self.nEpochs   = 1
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed            = 123
        self.upscale_factor  = 2
        self.training_loader = None
        self.testing_loader  = None

        self.save_path = None
        self.results   = collections.defaultdict(list)

        if not os.path.exists(f'{CHECKPOINT_DIR}'):
            os.makedirs(CHECKPOINT_DIR)

    def save_model(self, epoch, error_loss, test_loss):
        print(f"Error: {error_loss}, PSNR: {test_loss}")
        model_out_path = os.path.join(self.save_path,f"{test_loss}_{error_loss}_{epoch}.pth")
        torch.save(self.model, model_out_path)
        with open(os.path.join(self.save_path, '_log.csv'), 'a') as stream:
            stream.writelines(f"{self.config.model}, {self.optimizer.__class__.__name__}, {self.config.nEpochs}, {epoch}, {error_loss[0]}, {test_loss[0]}, {error_loss[1]}, {test_loss[1]}, {error_loss[2]}, {test_loss[2]}, "+os.linesep)
            stream.close()

    def checkpoint_path(self, _type=''):
        optimizer_state_doct = self.optimizer.state_dict()
        if len(optimizer_state_doct) > 0:
            self.save_path = f"{self.config.logprefix}_{self.config.model}_{self.optimizer.__class__.__name__}_{optimizer_state_doct['param_groups'][0]['lr']}"
        else:
            self.save_path = f"{self.config.logprefix}_{self.config.model}_{self.optimizer.__class__.__name__}"

        dest = os.path.join(CHECKPOINT_DIR,self.save_path).lower()

        try:
            if not os.path.exists(dest):
                os.makedirs(dest)
            else:
                print('Check point file deleting...')
                for itm in glob.glob(f"{dest}/*"):
                    os.remove(itm)
        except OSError as e:
            pass

        with open(os.path.join(dest, '_model.txt'), 'w+') as stream:
            stream.write(self.model.__str__())
            stream.close()
        with open(os.path.join(dest, '_optimizer.txt'), 'w+') as stream:
            stream.write(self.optimizer.__str__())
            stream.close()

        return dest

    def set_optimizer(self, _type='adam'):
        # Set parameters
        passed_type = _type
        _type = self.config.optim

        parameters = {}
        if passed_type == 'gan':
            pass
            parameters = {'params': self.netG.parameters(),'lr': self.lr}
        else:
            parameters = {'params': self.model.parameters(),'lr': self.lr}
        # if self.config.momentum > 0 and (_type not in ('sparseadam') or self.config.model not in ('edsr','drcn')):
            # parameters.update({'momentum': self.config.momentum})
        if self.config.weight_decay > 0 and _type not in ('sparseadam','rprop'):
            parameters.update({'weight_decay': self.config.weight_decay})

        # Set optimizar
        if _type == 'adam':
            self.optimizer = torch.optim.Adam(**parameters)
        elif _type == 'sparseadam':
            self.optimizer = torch.optim.SparseAdam(**parameters)
        elif _type == 'adamax':
            self.optimizer = torch.optim.Adamax(**parameters)
        elif _type == 'adam-gamma':
            self.optimizer = torch.optim.Adam(**parameters)
            # parameters.update({'eps': 1e-8})
            # parameters.update({'betas': (0.9, 0.999)})
            # self.optimizer = torch.optim.Adam(**parameters)
        elif _type == 'lamb':
            # parameters.update({'adam': ('adam' == 'adam')})
            parameters.update({'betas': (0.9, 0.999)})
            self.optimizer = Lamb(**parameters)
        elif _type == 'sgd':
            self.optimizer = torch.optim.SGD(**parameters)
        elif _type == 'asgd':
            self.optimizer = torch.optim.ASGD(**parameters)
        elif _type == 'adadelta':
            self.optimizer = torch.optim.Adadelta(**parameters)
        elif _type == 'adagrad':
            self.optimizer = torch.optim.Adagrad(**parameters)
        elif _type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(**parameters)
        elif _type == 'rprop':
            self.optimizer = torch.optim.Rprop(**parameters)

        # if method is GAN, dont call this schedule
        if _type != 'gan':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 750], gamma=0.1)

        self.save_path = self.checkpoint_path(_type=_type)

        return 

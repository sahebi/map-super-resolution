import argparse
import dataset.data as data
from torch.utils.data import DataLoader

from SRCNN.solver import SRCNNTrainer
from VDSR.solver import VDSRTrainer
from SubPixelCNN.solver import SubPixelTrainer
from MEMNET.solver import MEMNETTrainer
from FSRCNN.solver import FSRCNNTrainer
# from result.data import get_training_set, get_test_set

parser = argparse.ArgumentParser(description='Single Image Super Resolution train model')

# Model Parameter
parser.add_argument('--model', '-m',            type=str, default='srcnn',      help='choose which model is going to use (srcnn, vdsr, sub, mem, fsrcnn), Default=srcnn')
parser.add_argument('--dataset', '-ds',         type=str, default='BSDS300',    help='Name of dataset defined in dataset.yml, default=BSDS300')
parser.add_argument('--logprefix', '-l',        type=str, default='',           help='logfile prefix')
parser.add_argument('--patterb', '-p',          type=str, default='*.jpg',      help='Pattern of open file')
# Train Parameter
parser.add_argument('--batchSize', '-b',      type=int, default=16,             help='Training batch size')
parser.add_argument('--testBatchSize','-bt',  type=int, default=1,              help='Testing batch size')
parser.add_argument('--nEpochs','-n',         type=int, default=1,              help='Number of epochs to train for, Default=')
parser.add_argument('--lr',                   type=float, default=0.01,         help='Learning Rate. Default=0.01')
parser.add_argument('--seed',                 type=int, default=123,            help='random seed to use. Default=123')

parser.add_argument('--chanel',               type=int, default=3,              help='Color channel, Default=3')

# Optimization Parameter
parser.add_argument('--optim','-o',           type=str, default='adam',         help='Optimizer Name, sgd, adam, lamb')
parser.add_argument('--momentum',             type=float, default=0.9,          help='Momentum')
parser.add_argument('--weight_decay',         type=float, default=0.01,         help='Weight Decay')

# Optimization Scheduler
parser.add_argument('--schduler',             type=str, default='default',      help='Scheduler type')

# Disabled Parameters
parser.add_argument('--upscale_factor', '-uf',  type=int, default=1,            help="super resolution upscale factor, in this script this script is disabled")
args = parser.parse_args()

def main():
    print('Loading dataset')

    train_set            = data.get_data(dataset_name=args.dataset, data_type='train', upscale_factor=args.upscale_factor, pattern="*.16.lr.jpg")
    test_set             = data.get_data(dataset_name=args.dataset, data_type='test',  upscale_factor=args.upscale_factor, pattern="*.16.lr.jpg")

    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize,     shuffle=True)
    testing_data_loader  = DataLoader(dataset=test_set,  batch_size=args.testBatchSize, shuffle=False)

    models     = args.model.split(',')
    optimizers = args.optim.split(',')
    for i, args.model in enumerate(models):
        for j, args.optim in enumerate(optimizers):
            print(f"=============================================================")
            print(f"=====> {i+1}-{args.model}/{len(models)} {j+1}-{args.optim}/{len(optimizers)} training stated. reminder {(i+1)*(j+1)} / {(len(models) * len(optimizers))}")
            print(f"=============================================================")
            if args.model == 'srcnn':
                model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
            elif args.model == 'vdsr':
                model = VDSRTrainer(args, training_data_loader, testing_data_loader)
            elif args.model == 'sub':
                model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
            elif args.model == 'mem':
                model = MEMNETTrainer(args, training_data_loader, testing_data_loader)
            elif args.model == 'fsrcnn':
                model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
            model.run()

if __name__ =='__main__':
    main()
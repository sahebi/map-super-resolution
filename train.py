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
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m',            type=str, default='srcnn', help='choose which model is going to use')
parser.add_argument('--dataset', '-ds',         type=str, default='BSDS300', help='name of dataset defined in dataset.yml')
parser.add_argument('--logprefix', '-l',        type=str, default='', help='name of logfile')
# Train Parameter
parser.add_argument('--batchSize', '-b',      type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize','-bt',  type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs','-n',         type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr',                   type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed',                 type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--chanel',               type=int, default=3, help='random seed to use. Default=123')

# Optimization Parameter
parser.add_argument('--optim','-o',           type=str, default='adam', help='Optimizer Name, sgd, adam, lamb')
parser.add_argument('--momentum',             type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay',         type=float, default=0.01, help='Weight Decay')

# Optimization Scheduler
parser.add_argument('--schduler',             type=str, default='default', help='Scheduler type')
args = parser.parse_args()

def main():
    print('Loading dataset')

    train_set            = data.get_data(dataset_name=args.dataset, data_type='train', upscale_factor=args.upscale_factor)
    test_set             = data.get_data(dataset_name=args.dataset, data_type='test',  upscale_factor=args.upscale_factor)

    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader  = DataLoader(dataset=test_set,  batch_size=args.testBatchSize, shuffle=False)

    models     = args.model.split(',')
    optimizers = args.optim.split(',')
    for i, args.model in enumerate(models):
        for j, args.optim in enumerate(optimizers):
            print(f"====================================================")
            print(f"=====> {i+1}-{args.model}/{len(models)} {j+1}-{args.optim}/{len(optimizers)} training stated. reminder {(i+1)*(j+1)} / {(len(models) * len(optimizers))}")
            print(f"====================================================")
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

# Calculating FLops
# https://github.com/vra/flopth

# python train.py -l xxx -uf 1 -ds zurikh -b 4  -bt 8 -n 100 -m vdsr,sub,mem,fsrcnn,srcnn -o adam,sgd,lamb,adadelta,adagrad,rmsprop,rprop,adamax

# python train.py -l xxx -uf 1 -ds zurikh -b 4  -bt 8 -n 10 -m fsrcnn,srcnn -o sgd,adam,lamb,adadelta,adagrad,rmsprop,rprop,adamax

# python train.py -l xxx -uf 1 -ds zurikh -b 8  -bt 8 -n 10 -m srcnn
# python train.py -l xxx -uf 1 -ds zurikh -b 4  -bt 8 -n 10 -m fsrcnn
# python train.py -l xxx -uf 1 -ds zurikh -b 4  -bt 8 -n 10 -m vdsr
# python train.py -l xxx -uf 1 -ds zurikh -b 8  -bt 8 -n 10 -m sub
# python train.py -l xxx -uf 1 -ds zurikh -b 8  -bt 8 -n 10 -m mem
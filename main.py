import argparse, os
import pdb
import torch, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from medical_net import Xnet
from dataset import DatasetFromList

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Luna_X-Net')
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--step', type=int, default=15, help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=15')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--clip', type=float, default=10, help='Clipping Gradients. Default=10')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay, Default: 0')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

def main():

    global opt, model 
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print('===> Loading datasets')
    train_set = DatasetFromList('train_labels.txt',flag='train')
    test_set = DatasetFromList('test_labels.txt',flag='test')
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    print('===> Building model')
    model = Xnet()

    print('===> Setting GPU')
    if cuda:
        model = model.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)        
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)        
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
            
    print('===> Setting Optimizer')
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print('===> Training')
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, epoch)
        test(testing_data_loader, model)
        save_checkpoint(model, epoch)

def total_gradient(parameters):
    """Computes a gradient clipping coefficient based on gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters: 
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1./2)
    return totalnorm
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(training_data_loader, optimizer, model, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

    print 'epoch =', epoch,'lr =',optimizer.param_groups[0]['lr']
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1].type(torch.LongTensor), requires_grad=False)
        
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            
        loss = F.nll_loss(model(input), target)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        nn.utils.clip_grad_norm(model.parameters(),opt.clip)    

        optimizer.step()
        
        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
            print 'total gradient', total_gradient(model.parameters())
            
def test(testing_data_loader, model):
    test_loss = 0
    correct = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1].type(torch.LongTensor), requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        test_loss += F.nll_loss(prediction, target).data[0]
        pred = prediction.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testing_data_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testing_data_loader)*opt.testBatchSize,
        100. * correct / (len(testing_data_loader)*opt.testBatchSize)))

def save_checkpoint(model, epoch):
    model_folder = 'model/'
    model_out_path = model_folder + 'model_epoch_{}.pth'.format(epoch)
    state = {'epoch': epoch ,'model': model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
        
    print('Checkpoint saved to {}'.format(model_out_path))

if __name__ == '__main__':
    main()
import os
import torch


def evaluate_loss_accuracy(net, dataset, loss, device, adv=False):

    correct = 0.
    total = 0.
    total_loss = 0.

    with torch.no_grad():
        for x, y in dataset:
            if adv:
                x = torch.FloatTensor(x[1]).to(device)
            else:
                x = torch.FloatTensor(x).to(device)
            y = torch.LongTensor(y).to(device)

            pred = net(x) # pre-softmax
            total_loss += loss(pred, y).item()
            correct += (pred.argmax(dim=1) == y).sum()
            total += len(y)
        accuracy = correct / total
        total_loss /= len(dataset)

    return accuracy.item(), total_loss


def save_checkpoint(net, val_acc, adv_acc, epoch, logdir, model_string):
    """Saves model weights at a particular <epoch> into folder
    <logdir> with name <model_string>."""
    print('Saving..')
    """
    state = {
        'net': net,
        'val_acc': val_acc,
        'adv_acc': adv_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    """
    if not os.path.isdir(os.path.join(logdir, 'checkpoint/')):
        os.mkdir(os.path.join(logdir, 'checkpoint/'))

    # confirmed loading working in armory
    torch.save(net.state_dict(), os.path.join(logdir, 'checkpoint/') +
               model_string + '_epoch%d.pt' % epoch)
    
    
def adjust_learning_rate(optimizer, epoch, drop, base_learning_rate):
    """decrease the learning rate at <drop> epoch"""
    lr = base_learning_rate
    if epoch >= drop:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr    

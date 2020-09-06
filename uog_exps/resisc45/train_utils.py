import torch


def evaluate_loss_accuracy(net, dataset, loss, device):
    
    correct = 0.
    total = 0.
    total_loss = 0.

    with torch.no_grad():    
        for x, y in dataset:
            x = torch.FloatTensor(x).to(device)
            y = torch.LongTensor(y).to(device)
            
            pred = net(x) # pre-softmax
            total_loss += loss(pred, y).item()
            correct += (pred.argmax(dim=1) == y).sum()
            total += len(y)
        accuracy = correct / total
        total_loss /= len(dataset)
    
    return accuracy.item(), total_loss
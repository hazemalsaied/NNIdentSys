import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from config import configuration
from transitions import TransitionType

torch.manual_seed(1)


def getVocab(corpus):
    corpusSet = set()
    for s in corpus:
        tokens = [t.getTokenOrLemma() for t in s.tokens]
        sText = " ".join(tokens)
        corpusSet.update(sText)
    word_to_ix = {word: i for i, word in enumerate(corpusSet)}
    return word_to_ix


class PytorchModel(nn.Module):

    def __init__(self, normalizer):
        tokenEmb = configuration["model"]["embedding"]["tokenEmb"]
        posEmb = configuration["model"]["embedding"]["posEmb"]
        # tokenWeights = [normalizer.tokenWeightMatrix]
        # posWeights = [normalizer.posWeightMatrix]
        context_size = 4

        super(PytorchModel, self).__init__()
        self.tokenEmbeddings = nn.Embedding(len(normalizer.vocabulary.tokenIndices), tokenEmb)
        # , _weight=tokenWeights)
        self.posEmbeddings = nn.Embedding(len(normalizer.vocabulary.posIndices), posEmb)
        # , _weight=posWeights)
        self.linear1 = nn.Linear(context_size * (tokenEmb + posEmb), 24)
        self.linear2 = nn.Linear(24, len(TransitionType))

    def forward(self, tokenInputs, posInputs):
        tokenEmbeds = self.tokenEmbeddings(tokenInputs)  # .view((64, -1))
        posEmbeds = self.posEmbeddings(posInputs)  # .view((64, -1))
        out = F.relu(self.linear1(torch.cat((tokenEmbeds, posEmbeds), 2)))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def predict(self, trans, normalizer):
        tokenIdxs, posIdxs = normalizer.getAttachedIndices(trans)
        tokenItemsAsTensor = torch.tensor(tokenIdxs, dtype=torch.long)
        posItemsAsTensor = torch.tensor(posIdxs, dtype=torch.long)
        outputs = self.forward(tokenItemsAsTensor, posItemsAsTensor)
        _, sortedIndices = torch.sort(outputs[0], descending=True)
        # return sorted_scores, sorted_indices
        return sortedIndices

        # _, predicted = torch.max(outputs, 1)
        # return predicted


class TrainingDataSet(Dataset):

    def __init__(self, lbls, data):
        self.labels = lbls
        self.data = data
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


#
# def train(model, corpus, normalizer):
#     startTime = datetime.datetime.now()
#     losses, loss_function = [], nn.NLLLoss()
#     optimizer = optim.Adagrad(model.parameters(), lr=0.02)
#     labels, data = normalizer.generateLearningDataAttached(corpus)
#
#     dataIndices = range(len(labels))
#     for epoch in range(configuration["model"]["train"]["epochs"]):
#         random.shuffle(dataIndices)
#         sys.stdout.write('Epoch {0}'.format(epoch))
#         total_loss = torch.Tensor([0])
#         batchSize = configuration["model"]["train"]["batchSize"]
#         for i in range(0, len(labels), batchSize):
#             if i % 1000 == 0 and i != 0:
#                 sys.stdout.write('1000 batsh traited!')
#             batchTokenItems, batchPOSItems, targets = [], [], []
#             for j in range(batchSize):
#                 batchTokenItems.append(list(data[0][dataIndices[i + j]].flatten()))
#                 batchPOSItems.append(list(data[1][dataIndices[i + j]].flatten()))
#                 targets.append(labels[dataIndices[i + j]])
#
#             # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#             # into integer indices and wrap them in variables)
#             tokenItemsAsTensor = torch.tensor(batchTokenItems, dtype=torch.long)
#             posItemsAsTensor = torch.tensor(batchPOSItems, dtype=torch.long)
#
#             # Step 2. Recall that torch *accumulates* gradients. Before passing in a
#             # new instance, you need to zero out the gradients from the old
#             # instance
#             model.zero_grad()
#
#             # Step 3. Run the forward pass, getting log probabilities over next
#             # words
#             log_probs = model(tokenItemsAsTensor, posItemsAsTensor)
#
#             # Step 4. Compute your loss function. (Again, Torch wants the target
#             # word wrapped in a variable)
#             loss = loss_function(log_probs, torch.tensor(targets, dtype=torch.long))
#
#             # Step 5. Do the backward pass and update the gradient
#             loss.backward()
#             optimizer.step()
#
#             # Get the Python number from a 1-element Tensor by calling tensor.item()
#             total_loss += loss.item()
#         sys.stdout.write('Loss {0}'.format(total_loss))
#         losses.append(total_loss)
#     # for epoch in range(configuration["model"]["train"]["epochs"]):
#     #     sys.stdout.write('Epoch {0}'.format(epoch))
#     #     total_loss = torch.Tensor([0])
#     #     for i in range(len(labels)):
#     #         if i % 1000 == 0:
#     #             sys.stdout.write('1000 entries traited!')
#     #         tokenItems = list(data[0][i].flatten())
#     #         posItems = list(data[1][i].flatten())
#     #         target = [labels[i]]
#     #         # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#     #         # into integer indices and wrap them in variables)
#     #         tokenItemsAsTensor = torch.tensor(tokenItems, dtype=torch.long)
#     #         posItemsAsTensor = torch.tensor(posItems, dtype=torch.long)
#     #
#     #         # Step 2. Recall that torch *accumulates* gradients. Before passing in a
#     #         # new instance, you need to zero out the gradients from the old
#     #         # instance
#     #         model.zero_grad()
#     #
#     #         # Step 3. Run the forward pass, getting log probabilities over next
#     #         # words
#     #         log_probs = model(tokenItemsAsTensor, posItemsAsTensor)
#     #
#     #         # Step 4. Compute your loss function. (Again, Torch wants the target
#     #         # word wrapped in a variable)
#     #         loss = loss_function(log_probs, torch.tensor(target, dtype=torch.long))
#     #
#     #         # Step 5. Do the backward pass and update the gradient
#     #         loss.backward()
#     #         optimizer.step()
#     #
#     #         # Get the Python number from a 1-element Tensor by calling tensor.item()
#     #         total_loss += loss.item()
#     #     sys.stdout.write('Loss {0}'.format(total_loss))
#     #     losses.append(total_loss)
#     sys.stdout.write('# Training time = {0} minutes!\n'.format(str(datetime.now().minute - startTime.minute)))
#     return losses  # The loss decreased every iteration over the training data!
#
#
# from collections import OrderedDict
# def summary(input_size, model):
#     def register_hook(module):
#         def hook(module, input, output):
#             class_name = str(module.__class__).split('.')[-1].split("'")[0]
#             module_idx = len(summary)
#
#             m_key = '%s-%i' % (class_name, module_idx + 1)
#             summary[m_key] = OrderedDict()
#             summary[m_key]['input_shape'] = list(input[0].size())
#             summary[m_key]['input_shape'][0] = -1
#             summary[m_key]['output_shape'] = list(output.size())
#             summary[m_key]['output_shape'][0] = -1
#
#             params = 0
#             if hasattr(module, 'weight'):
#                 params += torch.prod(torch.LongTensor(list(module.weight.size())))
#                 if module.weight.requires_grad:
#                     summary[m_key]['trainable'] = True
#                 else:
#                     summary[m_key]['trainable'] = False
#             if hasattr(module, 'bias'):
#                 params += torch.prod(torch.LongTensor(list(module.bias.size())))
#             summary[m_key]['nb_params'] = params
#
#         if not isinstance(module, nn.Sequential) and \
#                 not isinstance(module, nn.ModuleList) and \
#                 not (module == model):
#             hooks.append(module.register_forward_hook(hook))
#
#     # check if there are multiple inputs to the network
#     if isinstance(input_size[0], (list, tuple)):
#         x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
#     else:
#         x = Variable(torch.rand(1, *input_size))
#
#     # create properties
#     summary = OrderedDict()
#     hooks = []
#     # register hook
#     model.apply(register_hook)
#     # make a forward pass
#     model(x)
#     # remove these hooks
#     for h in hooks:
#         h.remove()
#
#     return summary


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[0], data[1])
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(model, corpus, normalizer):
    # use_cuda = configuration["evaluation"]["cluster"]
    device = 'cpu' # torch.device("cuda" if use_cuda else "cpu")

    kwargs = {}# {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batchSize = configuration["model"]["train"]["batchSize"]
    epochs = configuration["model"]["train"]["epochs"]
    labels, data = normalizer.generateLearningDataAttached(corpus)
    trainingData = TrainingDataSet(labels, data)
    train_loader = torch.utils.data.DataLoader(
        trainingData,
        # datasets.MNIST('../data', train=True, download=True,
        #                transform=transforms.Compose([
        #                    transforms.ToTensor(),
        #                    transforms.Normalize((0.1307,), (0.3081,))
        #                ])),
        batch_size=batchSize, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=1, shuffle=True, **kwargs)

    # model = PytorchModel().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.02)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)


if __name__ == '__main__':
    t1 = torch.arange(6)
    t1 = torch.reshape(t1, (1, 2, 3))
    t2 = torch.arange(6)
    t2 = torch.reshape(t2, (1, 2, 3))
    print t1
    print t2
    print torch.cat((t1, t2), 0)

import os
import shutil
import time
from collections import Counter
from operator import itemgetter
from corpus import getTokens
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch import optim
# from torch.nn import Parameter
from torch.nn.init import *
from transitions import TransitionType
import decoder
from config import configuration

if 'GPU' not in os.environ or int(os.environ['GPU']) == 0:
    print 'Using CPU'
    use_gpu = False
else:
    print 'Using GPU'
    use_gpu = True

get_data = (lambda x: x.data.cpu()) if use_gpu else (lambda x: x.data)


def Variable(inner):
    return torch.autograd.Variable(inner.cuda() if use_gpu else inner)


def Parameter(shape=None, init=xavier_uniform):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def scalar(v):
    if type(v) == int:
        return Variable(torch.LongTensor([v]))
    if type(v) == float:
        return Variable(torch.FloatTensor([v]))


def cat(l, dimension=-1):
    return torch.cat(l, dimension)
    # valid_l = filter(lambda x: x, l)
    # if dimension < 0:
    #     dimension += len(valid_l[0].size())
    # return torch.cat(valid_l, dimension)


class RNNState:
    def __init__(self, cell, hidden=None):
        self.cell = cell
        self.hidden = hidden
        if not hidden:
            self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, i):
        return RNNState(self.cell, self.cell(i, self.hidden))

    def __call__(self):
        return self.hidden[0]


class MSTParserLSTMModel(nn.Module):
    def __init__(self, vocab, pos, w2i):
        super(MSTParserLSTMModel, self).__init__()
        random.seed(1)
        self.activations = {'tanh': f.tanh, 'sigmoid': f.sigmoid, 'relu': f.relu,
                            # Not yet supporting tanh3
                            # 'tanh3': (lambda x: nn.Tanh()(cwise_multiply(cwise_multiply(x, x), x)))
                            }
        self.activation = self.activations[configuration['kiperwasser']['activation']]

        self.blstmFlag = True
        # self.labelsFlag = configuration['kiperwasser']labelsFlag
        # self.costaugFlag = configuration['kiperwasser']costaugFlag
        self.bibiFlag = True  # configuration['kiperwasser']bibiFlag

        self.ldims = configuration['kiperwasser']['lstmUnitNum']
        self.wdims = configuration['kiperwasser']['wordDim']  # configuration['kiperwasser']wembedding_dims
        self.pdims = configuration['kiperwasser']['posDim']  # configuration['kiperwasser']pembedding_dims
        # self.rdims = configuration['kiperwasser']rembedding_dims
        self.layers = configuration['kiperwasser']['layerNum']  # configuration['kiperwasser']lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        # self.rels = {word: ind for ind, word in enumerate(rels)}
        # self.irels = rels

        self.edim = 0
        if self.bibiFlag:
            self.builders = [nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
                             nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]
            self.bbuilders = [nn.LSTMCell(self.ldims * 2, self.ldims),
                              nn.LSTMCell(self.ldims * 2, self.ldims)]
        elif self.layers > 0:
            assert self.layers == 1, 'Not yet support deep LSTM'
            self.builders = [
                nn.LSTMCell(self.wdims + self.pdims, self.ldims),
                nn.LSTMCell(self.wdims + self.pdims, self.ldims)]
        else:
            self.builders = [nn.RNNCell(self.wdims + self.pdims, self.ldims),
                             nn.RNNCell(self.wdims + self.pdims, self.ldims)]
        for i, b in enumerate(self.builders):
            self.add_module('builder%i' % i, b)
        if hasattr(self, 'bbuilders'):
            for i, b in enumerate(self.bbuilders):
                self.add_module('bbuilder%i' % i, b)
        self.hidden_units = configuration['kiperwasser']['dense1']
        self.hidden2_units = configuration['kiperwasser']['dense2']
        self.labelsFlag = False

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
        # self.rlookup = nn.Embedding(len(rels), self.rdims)

        self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units))
        self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units))
        self.hidBias = Parameter(self.hidden_units)

        if self.hidden2_units:
            self.hid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.hid2Bias = Parameter(self.hidden2_units)

        self.outLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1))

        if self.labelsFlag:
            self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units))
            self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units))
            self.rhidBias = Parameter(self.hidden_units)

            if self.hidden2_units:
                self.rhid2Layer = Parameter((self.hidden_units, self.hidden2_units))
                self.rhid2Bias = Parameter(self.hidden2_units)

            self.routLayer = Parameter(
                (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.irels)))
            self.routBias = Parameter((len(self.irels)))
        self.linear1 = nn.Linear(configuration['kiperwasser']['focusedElemNum'] *
                                 configuration['kiperwasser']['lstmUnitNum'] * 2, 128)
        # dropout here is very detrimental
        self.dropout1 = nn.Dropout(p=configuration['kiperwasser']['dropout'])
        self.linear2 = nn.Linear(128, len(TransitionType))
        self.lossFunction = nn.NLLLoss()


    def __getExpr(self, sentence, i, j, train):
        if sentence[i].headfov is None:
            sentence[i].headfov = torch.mm(cat([sentence[i].lstms[0], sentence[i].lstms[1]]),
                                           self.hidLayerFOH)
        if sentence[j].modfov is None:
            sentence[j].modfov = torch.mm(cat([sentence[j].lstms[0], sentence[j].lstms[1]]),
                                          self.hidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.outLayer,
                self.activation(
                    self.hid2Bias +
                    torch.mm(self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias),
                             self.hid2Layer)
                )
            )  # + self.outBias
        else:
            output = torch.mm(
                self.activation(
                    sentence[i].headfov + sentence[j].modfov + self.hidBias),
                self.outLayer)  # + self.outBias

        return output

    def __evaluate(self, sentence, t):
        exprs = [[self.__getExpr(sentence, i, j, t)
                  for j in xrange(len(sentence))]
                 for i in xrange(len(sentence))]
        scores = np.array([[get_data(output).numpy()[0, 0] for output in exprsRow] for exprsRow in exprs])
        return scores, exprs

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = torch.mm(cat([sentence[i].lstms[0], sentence[i].lstms[1]]),
                                            self.rhidLayerFOH)
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = torch.mm(cat([sentence[j].lstms[0], sentence[j].lstms[1]]),
                                           self.rhidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.routLayer,
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.rhid2Layer,
                        self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias)
                    ) +
                    self.routBias)
            )
        else:
            output = torch.mm(
                self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias),
                self.routLayer
            ) + self.routBias

        return get_data(output).numpy()[0], output[0]

    def predict(self, sentence):
        for entry in sentence:
            wordvec = self.wlookup(scalar(int(self.vocab.get(entry.norm, 0)))) if self.wdims > 0 else None
            posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            evec = self.elookup(scalar(int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0))))) \
                if self.external_embedding is not None else None
            entry.vec = cat([wordvec, posvec, evec])

            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

        if self.blstmFlag:
            lstm_forward = RNNState(self.builders[0])
            lstm_backward = RNNState(self.builders[1])

            for entry, rentry in zip(sentence, reversed(sentence)):
                lstm_forward = lstm_forward.next(entry.vec)
                lstm_backward = lstm_backward.next(rentry.vec)

                entry.lstms[1] = lstm_forward()
                rentry.lstms[0] = lstm_backward()

            if self.bibiFlag:
                for entry in sentence:
                    entry.vec = cat(entry.lstms)

                blstm_forward = RNNState(self.bbuilders[0])
                blstm_backward = RNNState(self.bbuilders[1])

                for entry, rentry in zip(sentence, reversed(sentence)):
                    blstm_forward = blstm_forward.next(entry.vec)
                    blstm_backward = blstm_backward.next(rentry.vec)

                    entry.lstms[1] = blstm_forward()
                    rentry.lstms[0] = blstm_backward()

        scores, exprs = self.__evaluate(sentence, True)
        heads = decoder.parse_proj(scores)

        for entry, head in zip(sentence, heads):
            entry.pred_parent_id = head
            entry.pred_relation = '_'

        if self.labelsFlag:
            for modifier, head in enumerate(heads[1:]):
                scores, exprs = self.__evaluateLabel(sentence, head, modifier + 1)
                sentence[modifier + 1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]



    def forward(self, sent, errs, lerrs):

        for token in sent.tokens:
            c = float(self.wordsCount.get(token.text.lower(), 0))
            dropFlag = (random.random() < (c / (0.25 + c)))
            wordvec = self.wlookup(scalar(
                int(self.vocab.get(token.text.lower(), 0)) if dropFlag else 0)) if self.wdims > 0 else None
            posvec = self.plookup(scalar(int(self.pos[token.posTag.lower()]))) if self.pdims > 0 else None
            # if self.external_embedding is not None:
            #     evec = self.elookup(scalar(self.extrnd.get(token.form, self.extrnd.get(token.norm, 0)) if (
            #             dropFlag or (random.random() < 0.5)) else 0))
            #     token.vec = cat([wordvec, posvec, evec])
            # else:
            #     token.vec = torch.cat([wordvec, posvec], -1)  # cat([wordvec, posvec])
            token.vec = torch.cat([wordvec, posvec], -1)  # cat([wordvec, posvec])
            token.lstms = [token.vec, token.vec]
            token.headfov = None
            token.modfov = None

            token.rheadfov = None
            token.rmodfov = None

        if self.blstmFlag:
            lstm_forward = RNNState(self.builders[0])
            lstm_backward = RNNState(self.builders[1])

            for token, rtoken in zip(sent.tokens, reversed(sent.tokens)):
                lstm_forward = lstm_forward.next(token.vec)
                lstm_backward = lstm_backward.next(rtoken.vec)

                token.lstms[1] = lstm_forward()
                rtoken.lstms[0] = lstm_backward()

            if self.bibiFlag:
                for token in sent.tokens:
                    token.vec = cat(token.lstms)

                blstm_forward = RNNState(self.bbuilders[0])
                blstm_backward = RNNState(self.bbuilders[1])

                for token, rtoken in zip(sent.tokens, reversed(sent.tokens)):
                    blstm_forward = blstm_forward.next(token.vec)
                    blstm_backward = blstm_backward.next(rtoken.vec)

                    token.lstms[1] = blstm_forward()
                    rtoken.lstms[0] = blstm_backward()
        #e = 0
        transLoss =[]
        trans = sent.initialTransition
        while trans and trans.next:
            y = trans.next.type.value
            # print("gold trans %s is to apply on config: %s" % (t, c))
            activeElemIdxs = getFocusedElems(trans.configuration)
            # prediction
            lstmVecs = []
            for idx in activeElemIdxs:
                for t in sent.tokens:
                    if t.position == idx:
                        lstmVecs.append(cat(t.lstms))
            lstmInput = cat(lstmVecs)
            out = f.relu(self.linear1(lstmInput))
            out = self.dropout1(out)
            out = self.linear2(out)
            y_pred_vec = f.log_softmax(out, dim=1)
            # loss
            loss = self.lossFunction(y_pred_vec,torch.LongTensor([y]).cuda() if use_gpu else torch.LongTensor([y]))
            #e += loss
            transLoss.append(loss)
            # epoch_loss += loss.data
            trans = trans.next
        e = sum(transLoss)

        scores, exprs = self.__evaluate(tokens, True)
        gold = [token.dependencyParent for token in tokens]
        heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

        if self.labelsFlag:
            for modifier, head in enumerate(gold[1:]):
                rscores, rexprs = self.__evaluateLabel(tokens, head, modifier + 1)
                goldLabelInd = self.rels[tokens[modifier + 1].relation]
                wrongLabelInd = \
                    max(((ll, scr) for ll, scr in enumerate(rscores) if ll != goldLabelInd), key=itemgetter(1))[0]
                if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                    lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        if e > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0] for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
        return e


def getFocusedElems(config):
    idxs = []
    if config.stack and len(config.stack) > 1:
        for t in getTokens(config.stack[-2])[:2]:
            idxs.append(t.position)
    while len(idxs) < 2:
        idxs = [-1] + idxs

    if config.stack:
        for t in getTokens(config.stack[-1])[:4]:
            idxs.append(t.position)
    while len(idxs) < 6:
        idxs = [-1] + idxs

    if config.buffer:
        for t in config.buffer[:2]:
            idxs.append(t.position)

    while len(idxs) < 8:
        idxs = [-1] + idxs

    return idxs

def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)


class MSTParserLSTM:
    def __init__(self, vocab, pos, w2i):
        model = MSTParserLSTMModel(vocab, pos, w2i)
        self.model = model.cuda() if use_gpu else model
        self.trainer = get_optim(configuration['kiperwasser']['optimizer'], self.model.parameters())

    def predict(self, corpus):
        for iSentence, sentence in enumerate(corpus.testingSents):
            self.model.predict(sentence.tokens)
            yield sentence.tokens

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def train(self, corpus):
        print torch.__version__
        batch = 1
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        iSentence = 0
        start = time.time()

        shuffledData = list(corpus.trainingSents)
        random.shuffle(shuffledData)
        errs = []
        lerrs = []
        for iSentence, sentence in enumerate(shuffledData):
            if iSentence % 100 == 0 and iSentence != 0:
                print 'Processing sentence number:', iSentence, \
                    'Loss:', eloss / etotal, \
                    'Errors:', (float(eerrors)) / etotal, \
                    'Time', time.time() - start
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0

            e = self.model.forward(sentence.tokens, errs, lerrs)
            eerrors += e
            eloss += e
            mloss += e
            etotal += len(sentence)
            if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                if len(errs) > 0 or len(lerrs) > 0:
                    eerrs = torch.sum(cat(errs + lerrs))
                    eerrs.backward()
                    self.trainer.step()
                    errs = []
                    lerrs = []
            self.trainer.zero_grad()
        if len(errs) > 0:
            eerrs = (torch.sum(errs + lerrs))
            eerrs.backward()
            self.trainer.step()
        self.trainer.zero_grad()
        print "Loss: ", mloss / iSentence


def train(corpus):
    wordCounter = Counter()
    posCounter = Counter()
    for s in corpus.trainingSents:
        for t in s.tokens:
            wordCounter.update({t.text.lower():1})
            posCounter.update({t.posTag.lower():1})
    words, w2i, pos = wordCounter, {w: i for i, w in enumerate(wordCounter.keys())}, posCounter.keys()

    parser = MSTParserLSTM(words, pos, w2i)
    for epoch in xrange(configuration['kiperwasser']['epochs']):
        print 'Starting epoch', epoch
        parser.train(corpus)
        # conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
        # devpath = os.path.join(options.output,
        #                        'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
        # utils.write_conll(devpath, parser.predict(options.conll_dev))
        # parser.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))
        #
        # if not conllu:
        #     os.system(
        #         'perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt')
        # else:
        #     os.system(
        #         'python src/utils/evaluation_script/conll17_ud_eval.py -v -w
        # src/utils/evaluation_script/weights.clas '
        # + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
        #     with open(devpath + '.txt', 'rb') as f:
        #         for l in f:
        #             if l.startswith('UAS'):
        #                 print 'UAS:%s' % l.strip().split()[-1]
        #             elif l.startswith('LAS'):
        #                 print 'LAS:%s' % l.strip().split()[-1]

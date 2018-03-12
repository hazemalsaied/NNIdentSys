import keras
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from corpus import getTokens
from model import Network, Normalizer
from vocabulary import empty
import settings


INPUT_WORDS = settings.PADDING_ON_S0 + settings.PADDING_ON_S1 + settings.PADDING_ON_B0


class NetworkMLPSimple(Network):
    def __init__(self, normalizer):

        if settings.USE_SEPERATED_EMB_MODULE:
            posLayer, posFlattenLayer = Network.createPOSEmbeddingModule(INPUT_WORDS, normalizer)
            tokenLayer, tokenFlattenLayer = Network.createTokenEmbeddingModule(INPUT_WORDS, normalizer)
            flattenLayer = keras.layers.concatenate([posFlattenLayer, tokenFlattenLayer])
        else:
            wordLayer, flattenLayer = Network.createEmbeddingModule(INPUT_WORDS, normalizer)
        # Auxiliary feature vectors
        auxFeatureLayer = Input(shape=(normalizer.nnExtractor.featureNum,), name='auxFeatureLayer')
        # Merge layer
        concLayer = keras.layers.concatenate([flattenLayer, auxFeatureLayer])
        # MLP module
        mainOutputLayer = self.createMLPModule(concLayer)
        if settings.USE_SEPERATED_EMB_MODULE:
            self.model = Model(inputs=[posLayer, tokenLayer, auxFeatureLayer], outputs=mainOutputLayer)
        else:
            self.model = Model(inputs=[wordLayer, auxFeatureLayer], outputs=mainOutputLayer)
        super(NetworkMLPSimple, self).__init__()

    def predict(self, trans, normalizer):
        dataEntry = normalizer.normalize(trans)
        inputVec = [np.asarray([dataEntry[0]]), np.asarray([dataEntry[1]])]
        oneHotRep = self.model.predict(inputVec, batch_size=1, verbose=settings.PREDICT_VERBOSE)
        return argmax(oneHotRep)


class NormalizerMLPSimple(Normalizer):
    """
        Reponsable for tranforming the (config, trans) into training data for the network training
    """

    def __init__(self, corpus):
        super(NormalizerMLPSimple, self).__init__(corpus, 2)

    # def normalize(self, trans):
    #     dataEntry1, dataEntry2, dataEntry3, dataEntry4 = [], [], [], []
    #     if trans.configuration.stack:
    #         dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
    #         if len(trans.configuration.stack) > 1:
    #             dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
    #     if trans.configuration.buffer:
    #         dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
    #     dataEntry4 = self.nnExtractor.vectorize(trans)
    #     emptyIdx = self.vocabulary.indices[empty]
    #
    #     dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=settings.PADDING_ON_S0, value=emptyIdx))[0]
    #     dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=settings.PADDING_ON_S1, value=emptyIdx))[0]
    #     dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=settings.PADDING_ON_B0, value=emptyIdx))[0]
    #
    #     dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
    #     return [dataEntry1, np.asarray(dataEntry4)]


    def normalize(self, trans, seperatedModules=False):
        dataEntry1, dataEntry2, dataEntry3, dataEntry4 = [], [], [], []
        if trans.configuration.stack:
            if seperatedModules:
                dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]),usePos=True)
                dataEntry11 = self.getIndices(getTokens(trans.configuration.stack[-1]), useToken=True)
            else:
                dataEntry1 = self.getIndices(getTokens(trans.configuration.stack[-1]))
            if len(trans.configuration.stack) > 1:
                if seperatedModules:
                    dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]),usePos=True)
                    dataEntry22 = self.getIndices(getTokens(trans.configuration.stack[-2]),useToken=True)
                else:
                    dataEntry2 = self.getIndices(getTokens(trans.configuration.stack[-2]))
        if trans.configuration.buffer:
            if seperatedModules:
                dataEntry3 = self.getIndices(trans.configuration.buffer[:2],usePos=True)
                dataEntry33 = self.getIndices(trans.configuration.buffer[:2],useToken=True)
            else:
                dataEntry3 = self.getIndices(trans.configuration.buffer[:2])
        dataEntry4 = self.nnExtractor.vectorize(trans)
        emptyIdx = self.vocabulary.indices[empty]
        if seperatedModules:
            dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=settings.PADDING_ON_S0, value=emptyIdx))[0]
            dataEntry11 = np.asarray(pad_sequences([dataEntry11], maxlen=settings.PADDING_ON_S0, value=emptyIdx))[0]
            dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=settings.PADDING_ON_S1, value=emptyIdx))[0]
            dataEntry22 = np.asarray(pad_sequences([dataEntry22], maxlen=settings.PADDING_ON_S1, value=emptyIdx))[0]
            dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=settings.PADDING_ON_B0, value=emptyIdx))[0]
            dataEntry33 = np.asarray(pad_sequences([dataEntry33], maxlen=settings.PADDING_ON_B0, value=emptyIdx))[0]

            dataEntry1 = np.concatenate((dataEntry1,dataEntry11, dataEntry2,dataEntry22, dataEntry3,dataEntry33), axis=0)
            return [dataEntry1, np.asarray(dataEntry4)]
        else:
            dataEntry1 = np.asarray(pad_sequences([dataEntry1], maxlen=settings.PADDING_ON_S0, value=emptyIdx))[0]
            dataEntry2 = np.asarray(pad_sequences([dataEntry2], maxlen=settings.PADDING_ON_S1, value=emptyIdx))[0]
            dataEntry3 = np.asarray(pad_sequences([dataEntry3], maxlen=settings.PADDING_ON_B0, value=emptyIdx))[0]

            dataEntry1 = np.concatenate((dataEntry1, dataEntry2, dataEntry3), axis=0)
            return [dataEntry1, np.asarray(dataEntry4)]

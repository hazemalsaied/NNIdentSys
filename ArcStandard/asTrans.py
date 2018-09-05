from enum import Enum


class Configuration:
    def __init__(self, stack, buffer, isInitial=False, ):
        self.buffer = buffer
        self.stack = stack
        self.isInitial = isInitial
        self.isTerminal = self.isTerminal()

    def isTerminal(self):
        if not self.buffer and not self.stack:
            return True
        return False

    def __str__(self):
        stackStr = printStack(self.stack)
        buffStr = '[ '
        if self.buffer:
            for elem in self.buffer[:2]:
                if elem != 0:
                    buffStr += elem.text + ','
                else:
                    buffStr += '0,'
            buffStr += ' ..' if len(self.buffer) > 2 else ''
        buffStr += ']'
        return 'S=' + stackStr + ' B=' + buffStr


class TransType(Enum):
    SHIFT = 0
    LEFT_ARC = 1
    RIGHT_ARC = 2


class Transition:

    def __init__(self, type, sent, previous, next, config=None, isInitial=False):
        if type:
            self.type = type
        self.previous = previous
        if previous:
            previous.next = self
        self.next = next
        self.sent = sent
        if isInitial:
            self.configuration = Configuration([0], sent.tokens, True)
            self.type = None
            sent.initialTransition = self
        else:
            self.configuration = config

    def isTerminal(self):
        if self.configuration.stack == [0] and not self.configuration.buffer:
            return True
        return False


def printStack(elemlist):
    stackStr = ''
    elemlistStrs = getStackElems(elemlist)
    for r in elemlistStrs:
        if r == '[' or r == ']':
            if stackStr.endswith(', '):
                stackStr = stackStr[:-2]
            stackStr += r
        else:
            stackStr += r + ', '
    return stackStr + ' ' * (25 - len(stackStr))


def getStackElems(elemlist):
    elemlistStrs = ['[']
    for elem in elemlist:
        if elem == 0:
            elemlistStrs.append('0')
        elif str(elem.__class__) == 'corpus.Token':
            elemlistStrs.append(elem.text)
        elif isinstance(elem, list) and len(elem) == 1 and isinstance(elem[0], list):
            elemlistStrs.extend(getStackElems(elem[0]))
        elif isinstance(elem, list) and len(elem):
            elemlistStrs.extend(getStackElems(elem))
    elemlistStrs.append(']')
    return elemlistStrs

# class Shift(Transition):
#
#     def __init__(self, sent, previous, next, config):
#         super(Shift, self).__init__(TransType.SHIFT, sent, previous, next, config)
#
#
# class LeftArc(Transition):
#
#     def __init__(self, sent, previous, next, config):
#         super(LeftArc, self).__init__(TransType.LEFT_ARC, sent, previous, next, config)
#
#
# class RightArc(Transition):
#
#     def __init__(self, sent, previous, next, config):
#         super(RightArc, self).__init__(TransType.RIGHT_ARC, sent, previous, next, config)

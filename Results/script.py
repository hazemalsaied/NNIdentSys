"""
    Used to copy golden test files and to merge dev with train cupt files in one file
    the result should be stored in Results/Gold to be used in the evaluation process
"""
import os

from shutil import copyfile

allSharedtask2Lang = ['BG', 'DE', 'EL', 'EN', 'ES', 'EU', 'FA', 'FR', 'HE', 'HI',
                      'HR', 'HU', 'IT', 'LT', 'PL', 'PT', 'RO', 'SL', 'TR']

path = '/Users/halsaied/PycharmProjects/NNIdenSys/ressources/sharedtask.2/'
newPath = '/Users/halsaied/PycharmProjects/NNIdenSys/Results/Gold'
for lang in allSharedtask2Lang:
    if not os.path.isdir(os.path.join(newPath, lang)):
        os.makedirs(os.path.join(newPath, lang))
    for root, dirs, filenames in os.walk(os.path.join(path, lang)):
        files = ['dev.cupt', 'train.cupt']
        if os.path.isfile(os.path.join(path, lang, 'dev.cupt')):
            with open(os.path.join(newPath, lang, 'train.dev.cupt'), 'w') as outfile:
                outfile.write('# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n')
                for fname in files:
                    with open(os.path.join(path, lang, fname)) as infile:
                        idx = 0
                        for line in infile:
                            if idx != 0:
                                outfile.write(line)
                            idx += 1
                        outfile.write('\n')
        else:
            copyfile(os.path.join(path, lang, 'train.cupt'), os.path.join(newPath, lang, 'train.dev.cupt'))
        for f in filenames:
            if f not in ['dev.cupt', 'test.cupt', 'train.cupt']:
                continue
            if f == 'test.cupt':
                copyfile(os.path.join(path, lang, f), os.path.join(newPath, lang, f))

import re
from jieba import lcut
"""
将每一个种不同的Subword Encoding解析为原本的Word-Level模式
"""


def word_cleaner(x):
    return x.strip('\r\n')


def enocde_from_seq(input, subword):
    if subword == 'char':
        return seq_to_char(input)
    elif subword == 'space':
        return seq_to_space(input)
    elif subword =='charcnn_en':
        return seq_to_charcnn(input)
    elif subword == 'charcnn_en2':
        tmp = input.strip('\r\n')
        words = tmp.split()
        return ' '.join(words)
    elif subword is None or subword=='':
        return input
    else:
        raise Exception('Subword is invalid')

def seq_to_char(input):
    tmp = input.strip('\r\n')
    words = tmp.split()
    res = []
    for word in words:
        res += list(word)
    return ' '.join(res)

def seq_to_space(input, space='<space>'):
    tmp = input.strip('\r\n')
    words = tmp.split()
    new_str = []
    for word in words:
        new_str += list(word)
        new_str.append(space)
    if len(new_str) > 0:
        new_str = new_str[0:-1]
    return ' '.join(new_str)

def seq_to_charcnn(input, space='<c_pad>', num=4):
    tmp = input.strip('\r\n')
    words = tmp.split()
    new_str = []
    for word in words:
        tmp = list(word)[0:4]
        tmp += ([space] * max(0, num-len(tmp)))
        assert len(tmp) == num
        new_str += tmp
    return ' '.join(new_str)


def revert_from_sentence(sentence, subword_option):
    sentence = sentence.replace('#', '')
    if subword_option == 'space' :
        sentence = revert_space(sentence)
    if subword_option == 'char' or subword_option == 'charcnn_en':
        sentence = revert_charlevel(sentence)
    elif subword_option == 'ghybrid':
        sentence = revert_ghybridlevel(sentence)
    elif subword_option == 'bpe':
        sentence = revert_bpe(sentence)
    elif subword_option == 'wpm':
        sentence = revert_wpm(sentence)
    else:
        sentence = word_cleaner(sentence)
    return sentence



def revert_space(input):
    assert type(input) is not list, 'input :%s' % input
    tmp = input.strip('\r\n')
    tmp = tmp.replace(' ', '')
    tmp = tmp.replace('<space>', ' ')
    return tmp

def revert_wpm(input):
    assert type(input) is not list, 'input :%s' % input
    tmp = input.strip('\r\n')
    tmp = tmp.replace(' ', '')
    tmp = tmp.replace('▁', ' ')
    tmp = tmp.replace('_', ' ')
    tmp = tmp.strip(' ')
    return tmp

def revert_bpe(input):
    assert type(input) is not list, 'input :%s' % input
    tmp = ' ' + input.strip('\r\n')
    tmp = tmp.replace(' @@ ', '')
    tmp = tmp.replace('@@ ', '')
    tmp = tmp.strip(' ')
    return tmp

def revert_ghybridlevel(input, B=' <B>', M=' <M> ', E=' <E> ' ):
    assert type(input) is not list, 'input :%s' % input
    tmp = ' ' + input.strip('\r\n')
    tmp = tmp.replace(B, '')
    tmp = tmp.replace(M, '')
    tmp = tmp.replace(E, '')
    tmp = tmp.strip(' ')
    return tmp

def revert_charlevel(input, space_token='<space>'):
    """
    引入后，再进行分词
    :param input:
    :param space_token:
    :return: word-level sequence
    """
    assert type(input) is not list, 'input :%s' % input
    tmp = ''.join(input.strip('\r\n').split())
    tmp = tmp.replace(space_token,' ')
    tmp = ' '.join(lcut(tmp))
    return tmp
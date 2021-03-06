import os
import sys

PHONEME2IDX = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ay': 6, 'b': 7, 'ch': 8, 'd': 9, 'dh': 10, 'eh': 11, 'er': 12, 'ey': 13, 'f': 14, 'g': 15,
               'hh': 16, 'ih': 17, 'iy': 18, 'jh': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'ng': 24, 'ow': 25, 'oy': 26, 'p': 27, 'pau': 28, 'r': 29, 's': 30, 'sh': 31, 'ssil':32, 
               't': 33, 'th': 34, 'uh': 35, 'uw': 36, 'v': 37, 'w': 38, 'y': 39, 'z': 40, 'zh': 41}

IDX2PHONEME = {0:'aa', 1:'ae', 2:'ah', 3:'ao', 4:'aw', 5:'ax', 6:'ay', 7:'b', 8:'ch', 9:'d', 10:'dh', 11:'eh', 12:'er', 13:'ey', 14:'f', 15:'g',
               16:'hh', 17:'ih', 18:'iy', 19:'jh', 20:'k', 21:'l', 22:'m', 23:'n', 24:'ng', 25:'ow', 26:'oy', 27:'p', 28:'pau', 29:'r', 30:'s', 31:'sh', 32:'ssil', 
               33:'t', 34:'th', 35:'uh', 36:'uw', 37:'v', 38:'w', 39:'y', 40:'z', 41:'zh'}

PHONEME_LIST = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g',
               'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'pau', 'r', 's', 'sh', 'ssil', 
               't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

PHONEME_LIST_WITH_BLANK = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g',
               'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'pau', 'r', 's', 'sh', 'ssil', 
               't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh', ' ']

PHONEME_MAP = [
    'a',  # "aa"
    'A',  # "ae"
    'h',  # "ah"
    'o',  # "ao"
    'w',  # "aw"
    'X',  # "ax"
    'y',  # "ay"
    'b',  # "b" | '0'
    'c',  # "ch" | '2'
    'd',  # "d" | '1'
    'D',  # "dh" | '7'
    'e',  # "eh"
    'r',  # "er"
    'E',  # "ey"
    'f',  # "f" | '3'
    'g',  # "g" | '5'
    'H',  # "hh"
    'i',  # "ih"
    'I',  # "iy"
    'j',  # "jh" | '2'
    'k',  # "k" | '5'
    'l',  # "l"
    'm',  # "m" | '0'
    'n',  # "n" | '1'
    'G',  # "ng" | '5'
    'O',  # "ow"
    'Y',  # "oy"
    'p',  # "p" | '0'
    '&',  # "pau"
    'R',  # "r"
    's',  # "s" | '6'
    'S',  # "sh" | '4'
    '.',  # "ssil"
    't',  # "t" | '1'
    'T',  # "th" | '7'
    'u',  # "uh"
    'U',  # "uw"
    'v',  # "v" | '3'
    'W',  # "w"
    '?',  # "y"
    'z',  # "z" | '6'
    'Z',  # "zh" | '4'
    ' ', # "BLANK"
]



#voicing unvoicing
# p b m
# t d n
# ch jh
# f v
# sh zh
# k g ng
# s z
# th dh


if __name__ == '__main__':
    print("len of PHONEME_MAP: ", len(PHONEME_MAP))
    print("len of PHONEME_LIST: ", len(PHONEME_LIST))
    print("len of PHONEME_LIST_WITH_BLANK: ", len(PHONEME_LIST_WITH_BLANK))
    #print(len(set(PHONEME_MAP)))



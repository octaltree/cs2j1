#!/usr/bin/env python3

import sys
import random
import numpy as np
import math
from functools import reduce
undefined = None

def main():
    strs = lambda xs: [str(x) for x in xs]
    join = lambda xs: ''.join(strs(xs))
    hm = HiddenMarkov()
    hm.read()
    task2(hm)
    return 0

def task1(hm):
    print(hm.generate(maxlen=0))
    print(hm.generate(maxlen=1))
    print(hm.generate(maxlen=2))
    print(hm.generate(maxlen=100))
    print(hm.generateGoaled(maxlen=1))
    print(hm.generateGoaled(maxlen=2))
    print(hm.generateGoaled(maxlen=100))

def task2(hm):
    xs = []
    while sum([len(x[0]) for x in xs]) < 1000:
        (string, sts) = hm.generateGoaled(100)
        pred = hm.viterbi(string)
        xs += [(string, sts, pred)]
    strs = lambda xs: [str(x) for x in xs]
    join = lambda xs: ''.join(strs(xs))
    print(' {0} \n{1}\n{2}'.format(xs[-1][0], join(xs[-1][1]), join(xs[-1][2])))
    num = sum([len(x[1]) - 2 for x in xs])
    numcorrect = sum([
        len([t for t in list(zip(*x[1:3]))[1:-1] if t[0] == t[1]])
        for x in xs])
    print('文字列1000以上での正答率 {}'.format(numcorrect/num))

def task3(hm):
    pass

class HiddenMarkov:
    __alphs = None
    __stnum = 0
    __states = None
    __delta = []
    getAlphs = lambda self: self.__alphs
    getStnum = lambda self: self.__stnum
    getStates = lambda self: self.__states
    getDelta = lambda self: self.__delta
    def __random(self, probabilities):
        rand = random.random() # 一様
        tmp = 0
        for i in range(len(probabilities)):
            tmp += probabilities[i]
            if rand < tmp:
                return i
        return 1000000000000000
    def __trans(self, frm):
        return self.__random(self.__delta[frm])
    def __getAlpha(self, st):
        return self.__alphs[self.__random(self.__states[st])]
    def generate(self, maxlen=128):
        currentst = 0
        res = ""
        sts = [0]
        for i in range(maxlen+1):
            if currentst != 0:
                sts += [currentst]
                if currentst == self.__stnum - 1:
                    break
                res += self.__getAlpha(currentst)
            currentst = self.__trans(currentst)
        return (res, sts)
    def generateGoaled(self, maxlen=128):
        while True:
            tmp = self.generate(maxlen=maxlen)
            if tmp[1][-1] == self.__stnum - 1:
                return tmp
    def read(self):
        # ダメならindex out of rangeで死んで頼む
        lines = sys.stdin.read().split('\n')
        statelines = []
        tmpdelta  = []
        for line in lines:
            if line.replace('\n', '').split(' ')[0] == 'sigma':
                self.__alphs = tuple(line.split(' ')[1:])
            elif line.replace('\n', '').split(' ')[0] == 'stnum':
                self.__stnum = int(line.split(' ')[1])
            elif line.replace('\n', '').split(' ')[0] == 'state':
                statelines += [line]
            elif line.replace('\n', '').split(' ')[0] == 'delta':
                splited = line.split(' ')
                tmpdelta += [
                        (int(splited[1]), int(splited[2]), float(splited[3]))]
        flat1 = lambda ls: sum(ls, [])
        sortedstate = sorted(statelines, key=lambda l: l.split(' ')[1])
        self.__states = tuple([None] +
                [tuple([float(c) for c in l.split(' ')[2:]]) for l in sortedstate]
                )
        n = self.__stnum
        self.__delta = [[0 for i in range(n)] for j in range(n)]
        for t in tmpdelta:
            self.__delta[t[0]][t[1]] = t[2]
    def debugVars(self):
        print(self.__dict__)
    def viterbi(self, string):
        v = Viterbi(self.__alphs, self.__stnum, self.__states, self.__delta)
        return v.predict(string)

class Viterbi:
    def __init__(self, alphs, stnum, states, delta):
        self.__alphs = alphs
        self.__stnum = stnum
        self.__states = states
        self.__delta = delta
    def __e(self, st, alph):
        return self.__states[st][''.join(self.__alphs).index(alph)]
    def __log(self, x):
        return -np.inf if x == 0 else math.log(x)
    def __fill(self, string):
        for idx in range(len(string)):
            for st in range(1, self.__stnum - 1):
                m = np.max([
                    self.__dp[j][idx] + self.__log(self.__delta[j][st])
                    for j in range(self.__stnum - 1)
                    ])
                self.__dp[st][idx+1] = self.__log(self.__e(st, string[idx])) + m
    def __read(self, length):
        current = self.__stnum - 1
        path = []
        for idx in reversed(range(length + 1)):
            path = [current] + path
            current = np.argmax([
                self.__dp[st][idx] + self.__delta[st][current]
                for st in range(self.__stnum - 1)
                ])
        path = [0] + path
        return path
    def predict(self, string):
        if self.__stnum == 2 or len(string) == 0:
            return [0, self.__stnum - 1]
        self.__dp = np.full((self.__stnum, len(string)+1), -np.inf)
        self.__dp[0][0] = 0
        self.__fill(string)
        return self.__read(len(string))

if __name__ == "__main__":
    exit(main())

# vim:fenc=utf-8 ff=unix ft=python ts=4 sw=4 sts=4 fdm=indent fdl=0 fdn=1:
# vim: si et cinw=if,elif,else,for,while,try,except,finally,def,class:

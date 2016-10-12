#!/usr/bin/env python3

import sys
import random
import numpy as np
import math
undefined = None

def main():
    hm = HiddenMarkov()
    hm.read()
    (string, sts) = hm.generate(100)
    strs = lambda xs: [str(x) for x in xs]
    print(string)
    print(''.join(strs(sts)))
    pred = hm.viterbi(string)
    print(''.join(strs(pred)))
    return 0

class HiddenMarkov:
    __alphs = None
    __stnum = 0
    __states = None
    __delta = []
    def __random(self, probabilities):
        rand = random.random() # 一様
        tmp = 0
        for i in range(len(probabilities)):
            tmp += probabilities[i]
            if rand < tmp:
                return i
        return -1
    def __trans(self, frm):
        return self.__random(self.__delta[frm])
    def __getAlpha(self, st):
        return self.__alphs[self.__random(self.__states[st])]
    def generate(self, maxlen=128):
        currentst = 0
        res = ""
        sts = []
        for i in range(maxlen+1):
            if currentst == self.__stnum - 1:
                break
            if currentst != 0:
                sts += [currentst]
                res += self.__getAlpha(currentst)
            currentst = self.__trans(currentst)
        return (res, sts)
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
    def predict(self, string):
        if self.__stnum == 0 or len(string) == 0:
            return []
        # 対数を取り確率1を0, -INFを1と表す
        self.__dp = np.ones((self.__stnum, len(string)))
        self.__dp[0][0] = 0
        for idx in range(1, len(string)):
            for st in range(1, self.__stnum - 1):
                e = lambda x, a: self.__states[x][''.join(self.__alphs).index(a)]
                m = max([self.__dp[j][st-1] * self.__delta[j][st] for j in range(self.__stnum)])
                self.__dp[st][idx] = e(st, string[idx]) * m
        print(self.__dp)
        return []

if __name__ == "__main__":
    exit(main())

# vim:fenc=utf-8 ff=unix ft=python ts=4 sw=4 sts=4 fdm=indent fdl=0 fdn=1:
# vim: si et cinw=if,elif,else,for,while,try,except,finally,def,class:

#!/usr/bin/env python3

import sys
import random
import numpy as np
import math
from fractions import Fraction
from functools import reduce
undefined = None

def main():
    strs = lambda xs: [str(x) for x in xs]
    join = lambda xs: ''.join(strs(xs))
    hm = HiddenMarkov()
    hm.read()
    task4(hm)
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
    ts = [hm.generate(100) for i in range(2000)]
    (states, delta) = Counter(hm).count(ts)
    print(diff((hm.getStates(), states), (hm.getDelta(), delta)))

def task4(hm):
    ss = [hm.generateGoaled(100)[0] for i in range(1000)] # :: [String]
    hs = [ViterbiDecoding().randomHm(hm.getStnum(), hm.getAlphs()).calc(ss)
            for i in range(100)] # :: [HiddenMarkov]
    es = [diff((hm.getStates(), i.getStates()), (hm.getDelta(), i.getDelta()))
            for i in hs] # :: [Num]
    print(min(es))

def diff(sts, ds):
    flat1 = lambda xs: sum(xs, []) # :: [[a]] -> [a]
    # squarezip :: [[a]] -> [[b]] -> [[(a, b)]]
    squarezip = lambda a, b: [list(zip(i[0], i[1])) for i in list(zip(a, b))]
    err = rss(
            flat1(squarezip(sts[0][1:], sts[1][1:])) +
            flat1(squarezip(ds[0], ds[1])))
    return err

def rss(xs): # :: [(Num, Num)] -> Num
    return sum([(i[0] - i[1]) ** 2 for i in xs])

class ViterbiDecoding:
    def __init__(self, hm=None):
        self.__hm = hm
        if self.__hm is not None:
            self.__stnum = self.__hm.getStnum()
            self.__alphs = self.__hm.getAlphs()
    def randomHm(self, stnum, alphs):
        self.__stnum = stnum
        self.__alphs = alphs
        firststates = [np.random.rand(len(alphs)) for i in range(stnum)]
        firststates[0] = None
        firststates[-1] = None
        firstdelta = np.zeros((stnum, stnum))
        self.__hm = self.__newhm(firststates, firstdelta)
        return self
    def __newhm(self, states, delta):
        return HiddenMarkov(self.__alphs, self.__stnum, states, delta)
    def calc(self, ss):
        prepres = None
        while True:
            pres = [self.__hm.viterbi(s) for s in ss]
            if (prepres is not None and
                    all([i[0] == i[1] for i in list(zip(pres, prepres))])):
                break
            ts = list(zip(ss, pres))
            (st, dl) = Counter(self.__hm).count(ts)
            self.__hm = self.__newhm(st, dl)
        return self.__hm

class Counter:
    def __init__(self, hm):
        self.__hm = hm
    def __obs(self):
        def res(tmp, t):
            (states, delta) = tmp
            (string, sts) = t
            for (st, a) in zip(sts[1:-1], string):
                states[st][''.join(self.__hm.getAlphs()).index(a)] += 1
            for (frm, to) in zip(sts[:-1], sts[1:]):
                delta[frm][to] +=  1
            return (states, delta)
        return res
    def __fmt(self, states, delta):
        def flt(x, xs):
            s = sum(xs)
            return x / s if s != 0 else 0
        finalstates = tuple([
            None if st is None else tuple([flt(a, st) for a in st])
            for st in states])
        finaldelta = [
            [flt(to, frm) for to in frm]
            for frm in delta]
        return (finalstates, finaldelta)
    def count(self, ts):
        firststates = [
                np.zeros(len(self.__hm.getAlphs()))
                for i in range(self.__hm.getStnum())]
        firststates[0] = None
        firststates[-1] = None
        firstdelta = np.zeros((self.__hm.getStnum(), self.__hm.getStnum()))
        return self.__fmt(*reduce(self.__obs(), ts, (firststates, firstdelta)))

class HiddenMarkov:
    __alphs = None # :: (char,)
    __stnum = 0 # :: int
    __states = None # :: ((float,),)
    __delta = [] # :: float[][]
    getAlphs = lambda self: self.__alphs
    getStnum = lambda self: self.__stnum
    getStates = lambda self: self.__states
    getDelta = lambda self: self.__delta
    def __init__(self, alphs=None, stnum=0, states=None, delta=None):
        self.__alphs = alphs
        self.__stnum = stnum
        self.__states = states
        self.__delta = delta
    def __random(self, probabilities):
        rand = random.random() # 一様
        tmp = 0
        for i in range(len(probabilities)):
            tmp += probabilities[i]
            if rand < tmp:
                return i
        raise Exception('確率の合計が1より小さい')
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
                self.__dp[st][idx] + self.__log(self.__delta[st][current])
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

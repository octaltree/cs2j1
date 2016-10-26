#!/usr/bin/env python3

import sys
import random
import numpy as np
import math
from fractions import Fraction
from functools import reduce
cimport numpy as np
DTYPE = np.float
ctypedef np.float64_t DTYPE_t

undefined = None

def task1(hm):
    print(hm.generate(maxlen=0))
    print(hm.generate(maxlen=1))
    print(hm.generate(maxlen=2))
    print(hm.generate(maxlen=100))
    print(hm.generateGoaled(maxlen=2))
    print(hm.generateGoaled(maxlen=50))

def task2(hm):
    xs = []
    while np.sum([len(string) for (string, sts, pred) in xs]) < 1000:
        (string, sts) = hm.generateGoaled(100)
        pred = Viterbi(hm).predict(string)
        xs.append((string, sts, pred))
    strs = lambda xs: [str(x) for x in xs]
    join = lambda xs: ''.join(strs(xs))
    print(' {0} \n{1}\n{2}'.format(xs[0][0], join(xs[0][1]), join(xs[0][2])))
    num = np.sum([len(sts) - 2 for (string, sts, pred) in xs])
    numcorrect = np.sum([
        len([t for t in list(zip(*x[1:3]))[1:-1] if t[0] == t[1]])
        for x in xs])
    print('文字列1000以上での正答率 {}'.format(numcorrect/num))

def task3(hm):
    ts = [hm.generate(100) for i in range(2000)]
    (states, delta) = Counter(hm).count(ts)
    print(diff((hm.getAlphabetProb(), states), (hm.getTransProb(), delta)))

def task4(hm):
    ss = [hm.generateGoaled(100)[0] for i in range(1000)] # :: [String]
    hs = [ViterbiDecoding().randomHm(hm.getNumStates(), hm.getAlphabets()).calc(ss)
            for i in range(100)] # :: [HiddenMarkov]
    es = [diff((hm.getAlphabetProb(), i.getAlphabetProb()), (hm.getTransProb(), i.getTransProb()))
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
    return np.sum([(i[0] - i[1]) ** 2 for i in xs])

class ViterbiDecoding:
    def __init__(self, hm=None):
        self.__hm = hm
        if self.__hm is not None:
            self.__stnum = self.__hm.getNumStates()
            self.__alphs = self.__hm.getAlphabets()
    def randomHm(self, stnum, alphs):
        self.__stnum = stnum
        self.__alphs = alphs
        # TODO 合計が1になるように
        firststates = [self.__random(len(alphs)) for i in range(stnum)]
        firststates[0] = None
        firststates[-1] = None
        firstdelta = [self.__random(stnum) for i in range(stnum)]
        for i in range(stnum):
            firstdelta[i][0] = 0
            firstdelta[stnum-1][i] = 0
        self.__hm = self.__newhm(firststates, firstdelta)
        return self
    def __random(self, n):
        # 合計が1以上なランダム列
        while True:
            tmp = np.random.rand(n)
            if np.sum(tmp) >= 1:
                return tmp
    def __newhm(self, states, delta):
        return HiddenMarkov(self.__stnum, self.__alphs, np.array(states), np.array(delta))
    def calc(self, ss):
        prepres = None
        while True:
            pres = [Viterbi(self.__hm).predict(s) for s in ss]
            print(pres)
            if (prepres is not None and
                    all([i[0] == i[1] for i in list(zip(pres, prepres))])):
                break
            prepres = pres
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
                states[st][''.join(self.__hm.getAlphabets()).index(a)] += 1
            for (frm, to) in zip(sts[:-1], sts[1:]):
                delta[frm][to] +=  1
            return (states, delta)
        return res
    def __fmt(self, states, delta):
        def flt(x, xs):
            s = np.sum(xs)
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
                np.zeros(len(self.__hm.getAlphabets()))
                for i in range(self.__hm.getNumStates())]
        firststates[0] = None
        firststates[-1] = None
        firstdelta = np.zeros((self.__hm.getNumStates(), self.__hm.getNumStates()))
        return self.__fmt(*reduce(self.__obs(), ts, (firststates, firstdelta)))

class HiddenMarkov:
    __alphabets = None # :: (char,)
    __numstates = 0 # :: int
    __alphabetprob = None # :: ((float,),)
    __transprob = [] # :: float[][]
    getAlphabets = lambda self: self.__alphabets
    getNumAlphabets = lambda self: len(self.__alphabets)
    getNumStates = lambda self: self.__numstates
    getAlphabetProb = lambda self: self.__alphabetprob
    getTransProb = lambda self: self.__transprob
    def __init__(self, numstates=None, alphabets=None,
            np.ndarray alphabetprob=None, np.ndarray transprob=None):
        self.__alphabets = alphabets
        self.__numstates = numstates
        self.__alphabetprob = alphabetprob
        self.__transprob = transprob
    def __random(self, probabilities):
        cdef double rand = random.random() # 一様
        cdef double tmp = 0
        for i in range(len(probabilities)):
            tmp += probabilities[i]
            if rand < tmp:
                return i
        raise Exception('確率の合計が1より小さい {}'.format(probabilities))
    def __trans(self, int frm):
        return self.__random(self.getTransProb()[frm])
    def __getAlph(self, int st):
        return self.getAlphabets()[self.__random(self.getAlphabetProb()[st])]
    def generate(self, int maxlen=128):
        cdef int currentst = 0
        string = ""
        sts = [0]
        for i in range(maxlen+1):
            if currentst != 0:
                sts.append(currentst)
                if currentst == self.getNumStates() - 1:
                    break
                string += self.__getAlph(currentst)
            currentst = self.__trans(currentst)
        return (string, sts)
    def generateGoaled(self, int maxlen=128):
        while True:
            tmp = self.generate(maxlen=maxlen)
            if tmp[1][-1] == self.getNumStates() - 1:
                return tmp
    def read(self):
        # ダメならindex out of rangeで死んで頼む
        lines = sys.stdin.read().split('\n')
        self.__numstates = int([
            l.split(' ')[1]
            for l in lines if l.split(' ')[0] == 'stnum'
            ][-1])
        self.__alphabets = [
            l.split(' ')[1:]
            for l in lines if l.split(' ')[0] == 'sigma'
            ][-1]
        self.__alphabetprob = np.zeros(
                (self.getNumStates(), self.getNumAlphabets()))
        self.__transprob = np.zeros(
                (self.getNumStates(), self.getNumStates()))
        for line in lines:
            cols = line.split(' ')
            if cols[0] == 'state':
                self.__alphabetprob[int(cols[1])][:] = np.array(cols[2:])
            elif cols[0] == 'delta':
                self.__transprob[int(cols[1])][int(cols[2])] = cols[3]
        #statelines = []
        #tmpdelta  = []
        #for line in lines:
        #    if line.replace('\n', '').split(' ')[0] == 'sigma':
        #        self.__alphabets = tuple(line.split(' ')[1:])
        #    elif line.replace('\n', '').split(' ')[0] == 'stnum':
        #        self.__numstates = int(line.split(' ')[1])
        #    elif line.replace('\n', '').split(' ')[0] == 'state':
        #        statelines += [line]
        #    elif line.replace('\n', '').split(' ')[0] == 'delta':
        #        splited = line.split(' ')
        #        tmpdelta += [
        #                (int(splited[1]), int(splited[2]), float(splited[3]))]
        #flat1 = lambda ls: np.sum(ls, [])
        #sortedstate = sorted(statelines, key=lambda l: l.split(' ')[1])
        #self.__alphabetprob = tuple([None] +
        #        [tuple([float(c) for c in l.split(' ')[2:]]) for l in sortedstate]
        #        )
        #n = self.__numstates
        #self.__transprob = [[0 for i in range(n)] for j in range(n)]
        #for t in tmpdelta:
        #    self.__transprob[t[0]][t[1]] = t[2]
    def debugVars(self):
        print(self.__dict__)

class Viterbi:
    def __init__(self, hm):
        self.__hm = hm
    def __e(self, int st, alph):
        return self.__hm.getAlphabetProb()[st][
                ''.join(self.__hm.getAlphabets()).index(alph)]
    def __log(self, x):
        return -np.inf if x == 0 else math.log(x)
    def __fill(self, string):
        for idx in range(len(string)):
            for st in range(1, self.__hm.getNumStates() - 1):
                self.__dp[st][idx+1] = (
                        self.__log(self.__e(st, string[idx])) +
                        np.max([
                            dp[idx] + self.__log(tr[st])
                            for (dp, tr) in zip(
                                self.__dp[:-1],
                                self.__hm.getTransProb()[:-1])]))
    def __read(self, int length):
        cdef int current = self.__hm.getNumStates() - 1
        path = []
        for idx in reversed(range(length + 1)):
            path = [current] + path
            current = np.argmax([
                dp[idx] + self.__log(tr[current])
                for (dp, tr) in zip(
                    self.__dp[:-1],
                    self.__hm.getTransProb()[:-1])])
        path = [0] + path
        return path
    def predict(self, string):
        if self.__hm.getNumStates() == 2 or len(string) == 0:
            return [0, self.__hm.getNumStates() - 1]
        self.__dp = np.full((self.__hm.getNumStates(), len(string)+1), -np.inf)
        self.__dp[0][0] = 0
        self.__fill(string)
        return self.__read(len(string))

# vim:fenc=utf-8 ff=unix ft=python ts=4 sw=4 sts=4 fdm=indent fdl=0 fdn=1:
# vim: si et cinw=if,elif,else,for,while,try,except,finally,def,class:

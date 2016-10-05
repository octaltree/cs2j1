#!/usr/bin/env python3

import sys
import random
undefined = None

def main():
    tmp = HiddenMarkov()
    tmp.read()
    print(tmp.generate(5))
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
        for i in range(maxlen+1):
            if currentst == self.__stnum - 1:
                break
            if currentst != 0:
                res += self.__getAlpha(currentst)
            currentst = self.__trans(currentst)
        return res
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

if __name__ == "__main__":
    exit(main())

# vim:fenc=utf-8 ff=unix ft=python ts=4 sw=4 sts=4 fdm=indent fdl=0 fdn=1:
# vim: si et cinw=if,elif,else,for,while,try,except,finally,def,class:

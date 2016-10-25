#!/usr/bin/env python
import bg
import sys

def main():
    hm = bg.HiddenMarkov()
    hm.read()
    if len(sys.argv) < 2:
        bg.task1(hm)
        bg.task2(hm)
        bg.task3(hm)
        bg.task4(hm)
    elif sys.argv[1] == "1":
        bg.task1(hm)
    elif sys.argv[1] == "2":
        bg.task2(hm)
    elif sys.argv[1] == "3":
        bg.task3(hm)
    elif sys.argv[1] == "4":
        bg.task4(hm)
    return 0

if __name__ == "__main__":
    exit(main())

import bg

def main():
    hm = bg.HiddenMarkov()
    hm.read()
    bg.task4(hm)
    return 0

if __name__ == "__main__":
    exit(main())

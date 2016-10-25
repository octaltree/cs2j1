all: dice seven

dice: bg.pyx
	./bg.pyx < ./dice.txt
seven: bg.pyx
	./bg.pyx < ./sevenstates.txt

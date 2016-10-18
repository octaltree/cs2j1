all: dice seven

dice: a.py
	./a.py < ./dice.txt
seven: a.py
	./a.py < ./sevenstates.txt

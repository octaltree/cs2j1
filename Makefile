all: compile

dice: compile
	python main.py < ./dice.txt

seven: compile
	python main.py < ./sevenstates.txt

compile: bg.pyx
	python setup.py build_ext --inplace

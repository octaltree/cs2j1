all: run

run: compile
	python main.py < ./dice.txt
	python main.py < ./sevenstates.txt

compile: bg.pyx
	python setup.py build_ext --inplace

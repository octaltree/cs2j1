all: output.pdf

output.pdf: output.tex
	platex output.tex && platex output.tex
	dvipdfmx output.dvi

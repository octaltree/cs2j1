#!/usr/bin/env zsh

venv="python3 -m venv"
pip="pip"

root=`git rev-parse --show-toplevel`
sandbox="${root}/ve"

${=venv} ${sandbox} &&\
source ${sandbox}/bin/activate &&\
${=pip} install --upgrade pip &&\
echo activated  &&\

${=pip} install numpy &&\

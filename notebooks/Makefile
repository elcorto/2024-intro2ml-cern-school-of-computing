pyfiles=$(shell ls -1 *.py | grep -E '[0-9]+_.+\.py')
notebooks=$(pyfiles:%.py=%.ipynb)

all: $(notebooks)

$(notebooks): %.ipynb: %.py
	jupytext --use-source-timestamp --update --to ipynb $<
	jupytext --use-source-timestamp --set-formats ipynb,py:percent $<

clean:
	rm -fv $(notebooks)

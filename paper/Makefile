.PHONY: pdf evidence

pdf:
	mkdir -p build
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf robogogogo.pdf

evidence:
	python3 scripts/collect_evidence.py --repo ..

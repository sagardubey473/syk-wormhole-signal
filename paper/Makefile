# Makefile for SYK traversable wormhole paper
# Usage:
#   make          - build figures and compile paper
#   make figures  - generate all figures
#   make paper    - compile LaTeX (assumes figures exist)
#   make clean    - remove build artifacts

LATEX = pdflatex
BIBTEX = bibtex
PYTHON = python3

SCRIPTS = scripts/fig1_signal_chaos.py \
          scripts/fig2_mu_mechanism.py \
          scripts/fig3_noise.py \
          scripts/fig4_disorder.py

FIGURES = figures/fig1_signal_chaos.pdf \
          figures/fig2_mu_mechanism.pdf \
          figures/fig3_noise.pdf \
          figures/fig4_disorder.pdf

.PHONY: all paper figures clean

all: figures paper

figures: $(FIGURES)

figures/fig1_signal_chaos.pdf: scripts/fig1_signal_chaos.py scripts/plot_style.py
	$(PYTHON) $<

figures/fig2_mu_mechanism.pdf: scripts/fig2_mu_mechanism.py scripts/plot_style.py
	$(PYTHON) $<

figures/fig3_noise.pdf: scripts/fig3_noise.py scripts/plot_style.py
	$(PYTHON) $<

figures/fig4_disorder.pdf: scripts/fig4_disorder.py scripts/plot_style.py
	$(PYTHON) $<

paper: main.pdf

main.pdf: main.tex references.bib $(FIGURES)
	$(LATEX) main
	$(BIBTEX) main
	$(LATEX) main
	$(LATEX) main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
	rm -f main.fls main.fdb_latexmk main.synctex.gz
	rm -f *.aux

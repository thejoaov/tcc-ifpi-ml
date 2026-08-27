FILE = documento
ARTIGO = artigo
OUTDIR = ./output

.PHONY: all pdf artigo pdfa docx roteiro miniaturas clean help

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: pdf artigo ## Gera a monografia estendida e o artigo para a BIA

pdf: ## Gera o PDF da monografia estendida
	@mkdir -p $(OUTDIR)
	-lualatex -shell-escape -interaction=nonstopmode $(FILE).tex
	-bibtex $(FILE).aux
	-lualatex -shell-escape -interaction=nonstopmode $(FILE).tex
	-lualatex -shell-escape -interaction=nonstopmode $(FILE).tex
	@mv $(FILE).pdf $(OUTDIR)/$(FILE).pdf
	@echo "PDF gerado em $(OUTDIR)/$(FILE).pdf"

artigo: ## Gera o PDF do artigo (depósito na BIA)
	@mkdir -p $(OUTDIR)
	-lualatex -shell-escape -interaction=nonstopmode $(ARTIGO).tex
	-bibtex $(ARTIGO).aux
	-lualatex -shell-escape -interaction=nonstopmode $(ARTIGO).tex
	-lualatex -shell-escape -interaction=nonstopmode $(ARTIGO).tex
	@mv $(ARTIGO).pdf $(OUTDIR)/$(ARTIGO).pdf
	@echo "PDF gerado em $(OUTDIR)/$(ARTIGO).pdf"

pdfa: artigo ## Converte o artigo para PDF/A com o nome exigido pela BIA
	gs -dPDFA=2 -dBATCH -dNOPAUSE -dNOOUTERSAVE \
		-sColorConversionStrategy=RGB \
		-sDEVICE=pdfwrite \
		-dPDFACompatibilityPolicy=1 \
		-sOutputFile=$(OUTDIR)/2026_tcc_jvsmendes.pdf \
		$(OUTDIR)/$(ARTIGO).pdf
	@echo "PDF/A gerado em $(OUTDIR)/2026_tcc_jvsmendes.pdf"

docx: pdf ## Gera DOCX a partir da monografia (requer pandoc)
	@mkdir -p $(OUTDIR)
	pandoc $(FILE).tex -o $(OUTDIR)/$(FILE).docx \
		--bibliography=bibliografia.bib \
		--citeproc \
		--resource-path=. \
		|| echo "AVISO: Falha ao gerar DOCX. Verifique se o pandoc está instalado."
	@echo "DOCX gerado em $(OUTDIR)/$(FILE).docx"

miniaturas:
	./roteiro-miniaturas.sh

roteiro:
	@mkdir -p $(OUTDIR)
	@test -d $(OUTDIR)/slides || $(MAKE) miniaturas
	pandoc $(OUTDIR)/mini-roteiro.md -o $(OUTDIR)/mini-roteiro.pdf \
		--pdf-engine=lualatex \
		--include-in-header=roteiro-estilo.tex \
		--lua-filter=roteiro-filtro.lua \
		-V fontsize=11pt \
		-V geometry:"a4paper,top=2cm,bottom=2cm,left=2.2cm,right=2.2cm" \
		-V colorlinks=true -V linkcolor=greendark -V urlcolor=greendark
	@echo "PDF gerado em $(OUTDIR)/mini-roteiro.pdf"

clean: ## Remove arquivos auxiliares do LaTeX
	rm -rf *.aux *.bbl *.toc *.out *.log *.nls *.nlo *.lof *.lot *.blg *.ilg *.idx *.ind *.lop *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml *.xdv *.nav *.snm mermaid

distclean: clean ## Remove PDFs locais gerados na raiz
	rm -f $(FILE).pdf $(ARTIGO).pdf

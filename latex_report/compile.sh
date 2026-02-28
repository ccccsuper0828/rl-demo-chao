#!/bin/bash
# LaTeX Compilation Script for Dino Jump Technical Report

echo "======================================"
echo "Compiling Dino Jump Technical Report"
echo "======================================"
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Please install LaTeX distribution:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-full"
    exit 1
fi

# Clean previous builds
echo "[1/4] Cleaning previous builds..."
rm -f main.aux main.log main.out main.toc main.pdf 2>/dev/null

# First pass
echo "[2/4] First compilation pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: First pass failed. Check main.log for details."
    exit 1
fi

# Second pass (for references and TOC)
echo "[3/4] Second compilation pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Third pass (ensure all references are resolved)
echo "[4/4] Final compilation pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Clean auxiliary files
echo "Cleaning auxiliary files..."
rm -f main.aux main.log main.out main.toc

# Check if PDF was created
if [ -f "main.pdf" ]; then
    echo ""
    echo "======================================"
    echo "SUCCESS! PDF generated: main.pdf"
    echo "======================================"
    echo ""
    ls -lh main.pdf
    echo ""
    echo "To view: open main.pdf"
else
    echo ""
    echo "======================================"
    echo "ERROR: PDF generation failed"
    echo "======================================"
    echo "Check main.log for error messages"
    exit 1
fi

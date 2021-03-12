#! /bin/sh
# script to clean up a directory of auxiliary files made by tex
file=$1
/bin/rm -f ${file}.dvi ${file}.log ${file}.aux ${file}.ps ${file}.idx
/bin/rm -f ${file}.*~ ${file}.lof ${file}.toc  ${file}.lot ${file}.nav
/bin/rm -f ${file}.snm ${file}.out ${file}.bbl ${file}.blg ${file}.vrb
/bin/rm -f ${file}.fls ${file}.fdb_latexmk
/bin/rm -rf ${file}_files
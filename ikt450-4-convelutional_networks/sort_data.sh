#!/bin/bash 
source_dir="./traning"

if [ ! $# -eq 0 ]; then
    if [[ -d $1 ]]; then
        source_dir=$1 
    else
        echo "Passed path is not a directory"
    fi
fi
echo "$source_dir"
cd "$source_dir" || exit

for f in *; do
    if [ -f "$f" ]; then
        prefix="${f%%_*}"
        if [ ! -d "${prefix}" ]; then
            mkdir "${prefix}"
        fi
        mv "${f}" "./${prefix}/"
    fi 
done

#!/usr/bin/env bash
cat semeval_data/all_text | hunspell -a | grep '^&' | cut -d' ' -f2,5- | sed 's/ /       /' | tr -d ' ' | sed 's/,/     /g' > semeval_data/hunspell_cache

#!/bin/sh

set -e

cargo doc -v --no-deps

# Manual doc deployment
echo "<meta http-equiv=refresh content=0;url=accurate/index.html>" >> target/doc/index.html
ghp-import -n target/doc
git push -f origin gh-pages

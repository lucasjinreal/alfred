python3 setup.py check

bumpver update -p -n

rm -r build/
rm -r dist/

python3 setup.py sdist
twine upload dist/*


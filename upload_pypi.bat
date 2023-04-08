python3 setup.py check

bumpver update -p -n

rm -r build/
rm -r dist/

python setup.py sdist
twine upload dist/*


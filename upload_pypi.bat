python3 setup.py check


rm -r build/
rm -r dist/

python setup.py sdist
twine upload dist/*


python3 setup.py check


rm  build/
rm dist/

python setup.py sdist
twine upload dist/*


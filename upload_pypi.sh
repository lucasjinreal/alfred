# check setup is correct or not
python setup.py check

sudo rm -r build/
sudo rm -r dist/

# pypi interface are not valid any longer
# python3 setup.py sdist
# python3 setup.py sdist upload -r pypi

# using twine instead
python3 setup.py sdist
twine upload dist/*

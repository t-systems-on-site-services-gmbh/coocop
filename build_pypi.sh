echo Please make sure to update the version number in setup.py and __init__.py
read -p "Press enter to continue"
rm -r dist
rm -r coocop.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*

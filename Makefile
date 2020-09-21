
default:
	python3 setup.py build
	python3 setup.py install --user

pypi:
	@echo \#Make a new tag
	@echo rm -rf dist/*
	@echo python3 setup.py sdist bdist_wheel
	@echo python3 -m twine upload dist/*
clean:
	python3 setup.py clean

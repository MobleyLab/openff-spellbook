
default:
	python3 setup.py build
	python3 setup.py install --user
clean:
	python3 setup.py clean

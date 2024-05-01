.PHONY: tests test

install:
	@pip install -r requirements-dev.txt

test:
	PYTHONPATH=. py.test --cov-report term-missing --cov=positional_vectorizer tests/ -s -vv --cov-fail-under=90

tests: test

lint:
	mypy positional_vectorizer --no-warn-no-return --ignore-missing-imports
	flake8 positional_vectorizer tests
	@black --check --exclude "(venv|notebooks)" . || ( echo "\n\n\tyou must run "make beautify"\n\n" && exit 1 )

beautify:
	black --exclude "(venv|notebooks)" .

build:
	pip install build
	echo $(VERSION) > VERSION
	python -m build -s
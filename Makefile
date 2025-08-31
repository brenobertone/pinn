all:
	@poetry run mypy .
	@poetry run autoflake --in-place --remove-unused-variables -r pinn/
	@poetry run black --line-length 79 .
	@poetry run isort .
	@poetry run flake8 .

run_training:
	poetry run python pinn/run_training.py

plot:
	poetry run python pinn/plot_by_config.py

show_config:
	poetry run python pinn/show_configs.py

zip_code:
	zip -r IC-PINN.zip pinn .gitignore Makefile poetry.lock pyproject.toml README.md setup.cfg
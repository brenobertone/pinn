.PHONY: help train visualize compare clean install test

# Default target
help:
	@echo "PINN Training & Visualization Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo ""
	@echo "Main Commands:"
	@echo "  make train            Interactive training (1D/2D problems)"
	@echo "  make visualize        Generate animations from results"
	@echo "  make compare          Compare experiment results"
	@echo ""
	@echo "Quick Tests:"
	@echo "  make test-1d          Quick test all 1D problems"
	@echo "  make test-compare     Compare slope limiters on 1D"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove pycache and temp files"
	@echo "  make clean-results    Remove all results (careful!)"
	@echo "  make results          Show experiment summary"
	@echo ""

# Install dependencies
install:
	poetry install

# Interactive training
train:
	@echo "Starting interactive training..."
	@poetry run python train_interactive.py

# Interactive visualization
visualize:
	@echo "Starting interactive visualization..."
	@poetry run python visualize_interactive.py

# Interactive comparison
compare:
	@echo "Starting experiment comparison..."
	@poetry run python compare_interactive.py

# Quick 1D test
test-1d:
	@echo "Quick test: All 1D problems"
	@poetry run python test_1d_quick.py

# Compare slope limiters
test-compare:
	@echo "Comparing slope limiters..."
	@poetry run python test_1d_compare_methods.py

# Show results summary
results:
	@echo "Experiment Results Summary"
	@echo "=========================="
	@poetry run python -c "from pinn.experiments.tracker import ExperimentTracker; \
		df = ExperimentTracker().load_experiments(); \
		print(f'\nTotal experiments: {len(df)}'); \
		print(f'Problems: {len(df[\"problem_name\"].unique())}'); \
		print(f'\nBest results:'); \
		print(df.groupby('problem_name')['final_loss'].min().to_string())" 2>/dev/null || echo "No results yet"

# Clean pycache
clean:
	@echo "Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done"

# Clean all results (dangerous!)
clean-results:
	@echo "WARNING: This will delete ALL results!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf results/ && echo "Results deleted" || echo "Cancelled"

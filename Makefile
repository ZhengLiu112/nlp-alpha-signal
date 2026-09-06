.PHONY: test test-all calibrate demo demo-null clean

test:            ## fast suite, skips the multi-seed calibration (~1 min)
	python scripts/run_tests.py --fast -v

test-all:        ## everything including calibration (~20 min)
	python scripts/run_tests.py -v

calibrate:       ## framework validation — run before trusting any real result
	python -c "import sys; sys.path.insert(0,'.'); \
	from alpha.research.calibration import calibrate_framework; \
	df, ok = calibrate_framework(n_seeds=10); \
	df.to_csv('results/calibration.csv', index=False); \
	sys.exit(0 if ok else 1)"

demo:            ## end-to-end study on synthetic data with a planted effect
	python scripts/run_study.py --synthetic

demo-null:       ## same pipeline on pure noise — should find nothing
	python scripts/run_study.py --synthetic-null

clean:
	rm -rf results/*.csv results/figures/* __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

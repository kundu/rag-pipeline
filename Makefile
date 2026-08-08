.PHONY: run run-auto run-embedding validate clean

run:
	python3 run_pipeline.py

run-auto:
	python3 run_pipeline.py --auto-approve

run-embedding:
	python3 run_pipeline.py --mode embedding --auto-approve

validate:
	python3 validate.py

clean:
	rm -f chunks.json index_metadata.json retrieval_results.json \
	      draft_answers.json review_overrides.json answer_audit.json \
	      revised_answers.json retrieval_metrics.json \
	      retrieval_error_analysis.json final_report.md \
	      llm_calls.jsonl pipeline_state.json

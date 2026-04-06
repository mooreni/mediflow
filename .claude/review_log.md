# Code Review Violation Log

One line per entry: short description of the problem and the fix. No duplicates — including entries that teach the same underlying principle, even if the surface details differ.

## Entries

- `visualize.py` `_doc_type_from_id`: missing docstring (purpose, args, return) — added full docstring per Code Standards rule.
- `visualize.py` `_aggregate`: `doc_id` accessed on the row without being in `required_keys`, giving an unhelpful bare `KeyError` — added `doc_id` to the validated keys so the error message states what failed, what the input was, and what was expected (Error Handling rule).
- `llm_judge.py` `EvaluationResult`: dataclass had no docstring — added class-level docstring describing purpose and attributes (Code Standards rule).
- `llm_judge.py` `_is_retryable`: one-liner docstring was missing Args and Returns sections — expanded to full docstring (Code Standards rule).
- `dataset.py`: missing module-level description comment — added top-level docstring per Code Standards rule.
- `dataset.py` `DatasetDoc`: dataclass had no docstring — added class-level docstring with Attributes section per Code Standards rule.
- `dataset.py` `_load_docs`: missing docstring (purpose, args, return) — added full docstring per Code Standards rule.
- `dataset.py` file-reading calls used bare `path.read_text` with no existence check, so failures produced unhelpful OS errors with no context — extracted `_read_doc` helper with a `FileNotFoundError` stating what failed and what was expected (Error Handling rule).
- `dataset.py` `load_eval_docs_by_type` and `load_summary_train_docs`: duplicated path-construction and DatasetDoc-building loops already present in `_load_docs` — unified through the new `_read_doc` helper (Maintainability/no-duplication rule).
- `s5_gemini_flash_dspy_predict.py` `translate`: called `get_translate_client()` inside the translate method, creating an I/O dependency inside business logic — moved client construction to `__init__` and stored as `self._token_client` (Design: keep I/O out of business logic; pass dependencies in).
- `s5_gemini_flash_dspy_predict.py` `translate`: `ValueError` message stated what failed and input doc_id but not what was expected — expanded message to include "expected a non-empty Russian string" (Error Handling: messages must state what failed, what the input was, and what was expected).
- `run_benchmark.py` `_build_scenarios`: bare `KeyError` raised on unknown scenario name with no context — added explicit guard raising `KeyError` stating what the bad input was and what was expected (Error Handling: messages must state what failed, what the input was, and what was expected).

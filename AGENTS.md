# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `molink/`: `entrypoints/` holds API and OpenAI-compatible servers, `engine/` orchestrates request handling, `comm/` manages DHT/pipeline metadata, `distributed/` tracks parallel state, `executor/` and `worker/` coordinate model execution, and `core/` customizes vLLM scheduling. Configuration helpers sit in `config.py`.
- Benchmarks and request generators are in `benchmark/`; assets for docs go in `resources/`.

## Build, Test, and Development Commands
- Create an env with Python 3.10+ and install dependencies: `pip install -e .` then `pip install grpcio-tools==1.71.0 protobuf==5.29.0` (keeps parity with vLLM deps).
- Run a basic API server: `python -m molink.entrypoints.api_server --model <hf_repo> --port 8080 --dtype=half --max_model_len 4096 --serving_layers 0,39`. Provide `--initial_peer <ip:port>` on follower nodes for multi-host runs.
- OpenAI-compatible server: `python -m molink.entrypoints.openai.api_server --model <hf_repo> --port 8080`.
- Simple load probe while a server runs: `python benchmark/base_multi_request_test.py` or `python benchmark/request_with_set_interval.py` to send concurrent `curl` requests.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indents, snake_case for functions/variables, PascalCase for classes. Keep type hints and minimal logging via `vllm.logger.init_logger`.
- Align new APIs with existing vLLM-style configs (e.g., dataclass configs, `verify_with_parallel_config`) and prefer explicit arguments over globals.
- Co-locate helper functions near their module (e.g., scheduler tweaks in `core/`, networking utilities in `comm/`).

## Testing Guidelines
- No formal suite exists yet; add `pytest` cases under `tests/` with filenames like `test_<module>.py` and model-light fixtures where possible.
- For distributed changes, include a smoke script (or reuse `benchmark/`) that documents the command line, model used, hardware, and expected log snippet; paste outputs in PRs.
- If adding async or scheduling logic, prefer unit tests that pin queue sizes or token counts to avoid flaky integration dependencies.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., “Add pipeline scheduler guard”). Keep body lines under 72 chars and mention related issues.
- In PRs include: summary of behavior change, commands run (`python -m molink.entrypoints...`, benchmarks), node layout (ports, `serving_layers`, `initial_peer`), and any API surface changes.
- Attach relevant logs or screenshots for load tests; note compatibility expectations with vLLM version in `requirements.txt`.

## Security & Configuration Tips
- Do not commit credentials or model tokens; pass addresses/peers via CLI flags or env vars. Avoid hard-coding private IPs in examples or defaults.
- Validate tensor/pipeline parallel settings against `MoLinkModelConfig.verify_with_parallel_config` to prevent mismatched head counts before deploying.

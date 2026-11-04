# Third-party notices

RATION includes code, prompts and data from the projects below. These parts keep their original
licenses. The full license texts are in [`licenses/`](licenses/).

| Project | License | Used in this repository |
| --- | --- | --- |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | MIT | Tool filter prompt in `baselines/toolfilter/core.py`; tool descriptions and schemas in `benchmarks/agentdojo/environments/*/tools.json` |
| [AgentDyn](https://github.com/SaFo-Lab/AgentDyn) | MIT | Tool descriptions and schemas in `benchmarks/agentdyn/environments/*/tools.json` |
| [ASB](https://github.com/agiresearch/ASB) | MIT | Task and tool data in `benchmarks/asb/upstream_data/` and `benchmarks/asb/environments/*/tools.json` |
| [AttriGuard](https://doi.org/10.5281/zenodo.20308739) | MIT | `baselines/attriguard/agentdojo/official.py`, `baselines/attriguard/asb/official/`, and parts of `baselines/attriguard/core.py` and `baselines/attriguard/prompts.py` |
| [DRIFT](https://github.com/SaFo-Lab/DRIFT) | MIT | `baselines/drift/agentdojo/official/`, `baselines/drift/asb/official/`, and the prompts in `baselines/drift/prompts.py` |
| [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) | MIT | DataSentinel known-answer instruction and response normalization in `baselines/datasentinel/core.py` |
| [DTap](https://github.com/AI-secure/DecodingTrust-Agent) | Apache-2.0 | Tool descriptions and task categories in `benchmarks/dtap/environments/`. They are extracted from DTap and changed to a different format. |
| [Progent](https://github.com/sunblaze-ucb/progent) | MIT | Always-allowed tool lists in `baselines/progent/agentdojo/config.py` |

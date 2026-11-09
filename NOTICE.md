# Provenance and notices

This pilot preserves upstream identifiers and pinned revisions in every task manifest. It is intended for local research evaluation.

- SWE-bench: https://github.com/SWE-bench/SWE-bench (MIT). The bundled Flask repository remains under Flask's BSD-3-Clause license.
- GAIA: https://arxiv.org/abs/2311.12983. Only the question printed in the paper is included; gated dataset splits are not redistributed.
- BrowseComp/simple-evals: https://github.com/openai/simple-evals (MIT).
- OSWorld: https://github.com/xlang-ai/OSWorld (Apache-2.0).
- JobBench dataset: https://huggingface.co/datasets/JobBench/job-bench (MIT).
- Mind2Web dataset: https://huggingface.co/datasets/osunlp/Mind2Web (CC BY 4.0); code is MIT.
- WebArena-Verified: https://github.com/ServiceNow/webarena-verified (Apache-2.0).

Upstream assets and source repositories may carry additional notices. Retain those notices when redistributing any materialized workspace.

## Harness implementation provenance

The provider/turn boundary and fresh-clone code-task lifecycle in
`src/mixed_pilot_harness/` were designed by adapting the corresponding
architecture and selected data-shape conventions from OpenSquilla 0.5.2
(source revision `c16be6206fa104c0a2f84d40d1f8097e7810e74f`,
https://gitee.com/blurdamon_admin/open-squilla.git, Apache-2.0; license copy at
`licenses/OpenSquilla-Apache-2.0.txt`). The implementation is intentionally reduced and does not include
OpenSquilla routing, aggregation, DAG, metaskill, gateway, memory, or iterative
verification subsystems.

## MTRouter collection extraction

`src/mixed_pilot_harness/collection/upstream/` contains selected sources from
[MTRouter](https://github.com/ZhangYiqun018/MTRouter), pinned to commit
`ee9e4573a128c3e54aa8c83800318bfb783c4dfe`. The upstream MIT license is retained
verbatim at `collection/upstream/LICENSE.md`, including its original copyright
notice. `collection/upstream/PROVENANCE.json` lists original paths, source hashes,
and local adaptations.

The local collection policy and loop adapt the upstream BaselineMode,
RouletteMode/propensity recording, Collector, and FlexibleAgent flow. They use
this repository's providers, explicit pricing, and native ScienceWorld splits.
This is a reduced collection port, not an unchanged reproduction of the paper's
six-model pool, training code, experimental splits, or complete HLE tool agent.

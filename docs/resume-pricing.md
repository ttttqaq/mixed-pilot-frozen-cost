# Resume with a new price snapshot or provider

`infer.py --resume` accepts a changed price table. Existing call costs are not
recomputed. Each new call records `cost_currency`, `pricing_snapshot`, and
`provider_snapshot`; new nodes record `config_snapshot`. Legacy calls without
these fields use the original `plan.json` snapshot. Recovery of a saved response
with missing cost uses its original price, never the new price.

Provider or currency changes require `--resume-partial`: unfinished jobs restart
from scratch and completed jobs are skipped. Historical attempts remain in the
ledger. The original `plan.json` stays unchanged; `resume-config.json` records the
latest resume configuration and events record configuration changes.

For multiple currencies, `summary.json.cost_by_currency` is the cumulative
breakdown. `total_api_cost`, `agent_cost`, `evaluation_cost`, and
`cost_by_profile` cover only the reported current `currency`. There is no implicit
FX conversion. Currency changes in budget-limited runs are rejected. Exported
traces retain their node's original configuration; individual calls carry the
authoritative price snapshot if prices changed during an unfinished trajectory.

For the existing GLM-4.7-Flash run moving from BigModel to OpenRouter, use the
original command with `--resume-partial` and
`--pricing configs/openrouter-glm-4.7-flash.prices.json`. Set
`OPENROUTER_API_KEY` in the launching terminal. Do not launch a second process
against an output directory already in use.

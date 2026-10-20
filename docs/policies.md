# Policy files

Every benchmark domain has four files that together define its risk policies.

| File | Contents |
|---|---|
| `adapters/<benchmark>/environments/<domain>/policy.yaml` | Tools, task categories, and the risk quantities each task category tracks |
| `adapters/<benchmark>/environments/<domain>/policies.<model>.json` | The constructed tool allow-lists and behavioral thresholds, one complete set per fold |
| `benchmarks/<benchmark>/environments/<domain>/folds.json` | The fold of every task |
| `benchmarks/<benchmark>/environments/<domain>/tools.json` | The declared tool interfaces (descriptions and argument schemas) |

The Provisioner and feedback prompts are shared by all domains of a benchmark:
`adapters/<benchmark>/config/provisioner.yaml` and `feedback.yaml`.

## `policy.yaml`

In the code, a task category is called a *scenario*.

```yaml
domain: agentdojo/banking
interface_artifacts:
  tools: ../../../../benchmarks/agentdojo/environments/banking/tools.json
adapter_artifacts:
  provisioner: ../../config/provisioner.yaml
  feedback: ../../config/feedback.yaml
description: >
  Personal online-banking assistant ...
defaults:
  grace_factor: 1.5

tools:
  get_balance:
    tags: [read]
  send_money:
    magnitude: [amount]
    tags: [write, transfer, destructive, irreversible]

whitelist: [get_iban, get_balance, get_most_recent_transactions]

scenarios:
  payments:
    description: >
      Move money: send, schedule, or adjust transactions.
    greylist: [send_money, schedule_transaction, update_scheduled_transaction]
    implicit_risks:
      global:
        - {id: total_tool_calls, kind: count}
        - {id: tokens_total, kind: tokens}
      scenario_specific:
        - {id: transfer_actions, kind: count, tools: "@transfer"}

fallback_scenario: full_access
provisioner_guidance: |
  ...
```

| Field | Meaning |
|---|---|
| `domain`, `description` | Shown to the Provisioner |
| `interface_artifacts.tools` | Path of `tools.json`; tool descriptions and argument types come from it |
| `adapter_artifacts` | Paths of the shared Provisioner prompt and feedback templates |
| `defaults.grace_factor` | Factor by which a behavioral threshold is increased after an intervention (1.5) |
| `tools.<name>.tags` | Labels used to select tools in a risk quantity, written as `@tag`. In AgentDyn and AgentDojo, `destructive` and `irreversible` tools are held once for confirmation |
| `tools.<name>.magnitude` | Numeric arguments that per-action and total limits may bind to |
| `tools.<name>.targets` | Arguments naming a destination (recipient, URL, account) that value constraints may bind to |
| `tools.<name>.strip_prefix` | Prefix the tool itself strips from an argument (e.g. `#` in a channel name), applied before comparing values |
| `whitelist` | Read tools available in every task category |
| `scenarios.<id>.description` | The task category description the Provisioner selects from |
| `scenarios.<id>.greylist` | Additional tools the task category may use |
| `scenarios.<id>.implicit_risks` | Risk quantities tracked for the task category, in two tiers: `global` and `scenario_specific` |
| `fallback_scenario` | The fallback category, used when no other task category matches the request |
| `provisioner_guidance` | Domain-specific instructions added to the Provisioner prompts |
| `resolvers` | (DTap customer-service only) tools whose outputs supply values that satisfy a value constraint, e.g. an order id that maps to the customer the user named |

A risk quantity has these fields:

| Field | Meaning |
|---|---|
| `id` | Name; the key of its threshold in `policies.<model>.json` |
| `kind` | `count` (number of calls), `param_sum` (sum of a numeric argument), `param_each` (largest single value of a numeric argument), or `tokens` (token usage) |
| `tools` | Tool names or `@tag` selectors; empty means every tool |
| `parameter` | The argument for `param_sum` and `param_each` |
| `severity` | `soft` (default). Explicit limits from the user request are always `hard` |
| `grace_factor` | Overrides `defaults.grace_factor` for this quantity |
| `description` | Shown to the agent in feedback messages |

## `policies.<model>.json`

```json
{
  "k": 5,
  "quantile": 0.95,
  "folds": [
    {
      "payments": {
        "allowed_tools": ["get_iban", "get_balance", "send_money"],
        "limits": {"total_tool_calls": 7.0, "tokens_total": 11604.0, "transfer_actions": 1.0}
      }
    }
  ]
}
```

`folds` holds `k` complete sets of implicit risk policies, one per fold. Each set gives every task
category its tool allow-list (`allowed_tools`) and a behavioral threshold (`limits`) for every risk
quantity declared in `policy.yaml`. They replace the task category's tools and thresholds at load
time (`DomainTable.with_policy`). Fold *i* was constructed from the historical executions of the
tasks in the other four folds, and it governs the tasks whose fold in `folds.json` is *i*.

## `folds.json`

```json
{"k": 5, "folds": {"user_task_0": 1, "user_task_1": 4}}
```

Keys are the benchmark's task identifiers: user tasks for AgentDyn and AgentDojo, attacker tools
for ASB, and task ids for DTap.

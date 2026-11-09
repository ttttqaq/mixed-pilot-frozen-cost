"""Round-based online policy improvement with paired, verified interventions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

from .data import assert_training_rows
from .learning import LearningSystem, state_key
from .runtime import Ledger, Runner, BudgetExceeded, UnknownCostError, RecoverableRequestError, InvalidTeacherEvidenceError, _atomic
from .teacher import FixedReferenceTeacher, append_jsonl


def utility(value, config):
    return value["reward"] - config["budget"]["lambda_"] * value["cost"] / config["budget"]["cost_reference"]


def utility_std(value, config):
    # Ensemble disagreement is an acquisition heuristic, not a calibrated CI.
    return value.get("reward_std", 0.) + config["budget"]["lambda_"] * value.get("cost_std", 0.) / config["budget"]["cost_reference"]


def result_target(result, start, tail=None):
    """Actual suffix cost + estimated tail; inherited prefix/assessor excluded."""
    cost = sum(s["cost"] for s in result["all_steps"][start:])
    if result["terminal"] and result["labels_valid"]:
        return {"reward": result["reward"], "cost": cost, "reward_std": 0., "cost_std": 0., "source": "real", "ready": True}
    if result["status"] != "truncated" or tail is None or not tail.get("ready"):
        return None
    return {"reward": tail["reward"], "cost": cost + tail["cost"],
            "reward_std": 0., "cost_std": 0.,
            "source": "llm_teacher", "ready": True}


class FrozenPolicy:
    """Weights stay unchanged until the round is collected; epsilon is explicit."""
    def __init__(self, learner, config, version, rng, *, greedy=False, output=None, epsilon=None):
        self.learner, self.config, self.version, self.rng = learner, config, version, rng
        self.epsilon = epsilon if epsilon is not None else 0. if greedy else config["online"]["exploration_epsilon"]
        self.output = output

    def __call__(self, state):
        values = self.learner.predict_actions(state)
        pool = self.config["model_pool"]
        best = max(pool, key=lambda name: utility(values[name], self.config))
        probabilities = {name: self.epsilon / len(pool) + (1 - self.epsilon if name == best else 0.) for name in pool}
        selected = self.rng.choice(pool) if self.epsilon > 0 and self.rng.random() < self.epsilon else best
        if self.output is not None:
            append_jsonl(self.output / "policy_decisions.jsonl", {"snapshot_id": state.get("snapshot_id"),
                "task_id": state["task_id"], "split": state["split"], "policy_version": self.version,
                "selected_model": selected, "selection_probability": probabilities[selected],
                "probabilities": probabilities, "values": values, "epsilon": self.epsilon})
        return selected

    def describe(self, state):
        values = self.learner.predict_actions(state)
        pool = self.config["model_pool"]
        best = max(pool, key=lambda name: utility(values[name], self.config))
        return {"version": self.version, "rule": "epsilon-greedy frozen Router; recomputed at each future state",
                "epsilon": self.epsilon, "models": pool,
                "endpoint_selection_probabilities": {n: self.epsilon / len(pool) + (1-self.epsilon if n == best else 0.) for n in pool},
                "lambda": self.config["budget"]["lambda_"], "cost_reference": self.config["budget"]["cost_reference"]}


class OnlineTrainer:
    learner_class = LearningSystem
    runner_class = Runner
    ledger_class = Ledger

    def __init__(self, config, train, valid, manifest, harness, pricing, *, resume=False,
                 episode_factory=None, provider_factory=None, learner=None, simulation=False, references=None):
        self.config, self.train, self.valid, self.manifest = config, train, valid, manifest
        self.eval_split = config["dataset"]["validation_split"]
        self.out = Path(config["output_dir"])
        self._run_lock = None
        if not simulation:
            import fcntl
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self._run_lock = self.out.with_name(self.out.name + ".lock").open("a+")
            try:
                fcntl.flock(self._run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._run_lock.close()
                raise ValueError("Another training process owns this output directory")
        self.resume = resume
        self.simulation = simulation
        self.rng = random.Random(config["seed"])
        self.router_rows, self.pairs = {}, {}
        self.completed_iterations = 0
        self.cold_done = False
        self.best_valid = None
        self.last_evaluated = None
        self.cold_stage = None
        self.cold_query_refs = None
        self.cold_queries_done = 0
        if resume:
            previous = json.loads((self.out / "config.resolved.json").read_text())
            if previous != config:
                raise ValueError("Resume requires identical configuration and the reference-router-v2 format")
            saved_manifest = json.loads((self.out / "split_manifest.json").read_text())
            if saved_manifest["split_fingerprint"] != manifest["split_fingerprint"]:
                raise ValueError("Selected train/valid data changed since checkpoint")
            if saved_manifest.get("runtime_fingerprint") != manifest.get("runtime_fingerprint"):
                raise ValueError("Runtime source/provider/pricing changed since checkpoint")
            if saved_manifest.get("reference_fingerprint") != manifest.get("reference_fingerprint"):
                raise ValueError("Train references changed since checkpoint")
            state = json.loads((self.out / "state.json").read_text())
            self.completed_iterations = state["completed_iterations"]
            self.cold_done = state["cold_done"]
            self.best_valid = state.get("best_valid")
            self.last_evaluated = state.get("last_evaluated")
            self.cold_stage = state.get("cold_stage")
            self.cold_query_refs = state.get("cold_query_refs")
            self.cold_queries_done = state.get("cold_queries_done", 0)
            self.router_rows = {r["sample_id"]: r for r in state.get("router_replay", [])}
            self.pairs = {r["pair_id"]: r for r in state.get("pairs", [])}
            def tuples(x):
                return tuple(tuples(v) for v in x) if isinstance(x, list) else x
            self.rng.setstate(tuples(state["rng_state"]))
            self.learner = learner or self.learner_class.load(self.out / state["checkpoint"])
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            _atomic(self.out / "config.resolved.json", config)
            _atomic(self.out / "split_manifest.json", manifest)
            self.learner = learner or self.learner_class(config)
        self.ledger = self.ledger_class(self.out, config["budget"]["total"], pricing["currency"])
        self.runner = self.runner_class(config, harness, pricing, self.ledger, episode_factory=episode_factory,
                             provider_factory=provider_factory)
        self.evidence = FixedReferenceTeacher(config, self.runner, self.out, references or {})
        self.task_map = {t["task_id"]: t for t in train}
        self.teacher_predictions = {}
        if resume and self.runner.resilience.get("enabled", False):
            self.runner.recover_uncertain_calls()

    def event(self, kind, **values):
        append_jsonl(self.out / "events.jsonl", {"event": kind, "time": time.time(), **values})
        print(json.dumps({"event": kind, **values}, ensure_ascii=False), flush=True)

    def _policy(self, version, *, greedy=False):
        return FrozenPolicy(self.learner, self.config, version, self.rng, greedy=greedy, output=self.out)

    def _policy_id(self, *, greedy=False):
        return self.learner.policy_fingerprint(0. if greedy else self.config["online"]["exploration_epsilon"])

    def _save(self, completed, *, cold_done=None):
        if cold_done is not None:
            self.cold_done = cold_done
        self.completed_iterations = completed
        # Checkpoint and replay are a generation: state.json is the commit point.
        name = "checkpoints/" + uuid.uuid4().hex + ".pt"
        self.learner.save(self.out / name, {"completed_iterations": completed, "cold_done": self.cold_done,
            "split_fingerprint": self.manifest["split_fingerprint"], "simulation": self.simulation,
            "router_policy_version": self._policy_id()})
        _atomic(self.out / "state.json", {"completed_iterations": completed, "cold_done": self.cold_done,
            "cold_stage": self.cold_stage, "cold_query_refs": self.cold_query_refs,
            "cold_queries_done": self.cold_queries_done,
            "checkpoint": name, "best_valid": self.best_valid, "last_evaluated": self.last_evaluated, "rng_state": self.rng.getstate(),
            "router_replay": list(self.router_rows.values()),
            "pairs": list(self.pairs.values())})
        # latest.pt is a convenient deployment copy; resume uses the generation.
        self.learner.save(self.out / "latest.pt", {"completed_iterations": completed, "simulation": self.simulation,
            "router_policy_version": self._policy_id()})

    def _put_router(self, row):
        assert_training_rows([row])
        previous = self.router_rows.get(row["sample_id"])
        if previous == row:
            return
        if previous and previous["source"] == "real" and row["source"] != "real":
            return
        self.router_rows[row["sample_id"]] = row
        append_jsonl(self.out / "router_samples.jsonl", row)

    async def _real_rows(self, result, *, start=0):
        if not result["terminal"] or not result["labels_valid"] or result["split"] != "train":
            return
        steps = result["all_steps"]
        remaining = [0.] * len(steps)
        cost = 0.
        for i in reversed(range(len(steps))):
            cost += steps[i]["cost"]
            remaining[i] = cost
        for i in range(start, len(steps)):
            s = steps[i]
            self._put_router({"sample_id": s["call_id"], "state": s["before_state"], "model": s["profile"],
                "reward": result["reward"], "cost": remaining[i], "source": "real", "weight": 1.,
                "policy_version": result["policy_version"], "continuation": "frozen_epsilon_policy",
                "trajectory_id": result["trajectory_id"]})
    def _visits(self, task, result):
        return [{"task": task, "state": step["before_state"], "prefix": result["all_steps"][:i],
                 "trajectory_id": result["trajectory_id"]}
                for i, step in enumerate(result["all_steps"]) if step.get("transition") is not None]

    def _select(self, visits, count):
        # One pool entry per exact observed state; a long trace cannot flood it
        # with duplicate references to a shared prefix.
        unique = {}
        for visit in visits:
            unique[(visit["state"]["task_id"], state_key(visit["state"]))] = visit
        pool = list(unique.values())
        if not self.config["selection"]["enabled"]:
            return self.rng.sample(pool, min(count, len(pool)))
        for visit in pool:
            values = list(self.learner.predict_actions(visit["state"]).values())
            gap = abs(utility(values[0], self.config) - utility(values[1], self.config))
            uncertainty = sum(utility_std(v, self.config) for v in values)
            impact = abs(values[0]["reward"] - values[1]["reward"]) + .05
            query_cost = max(.001, sum(v["cost"] for v in values))
            cfg = self.config["selection"]
            visit["acquisition"] = ((cfg["impact_weight"] * impact + cfg["uncertainty_weight"] * uncertainty)
                                    * (uncertainty + .01) / (gap + uncertainty + .01)
                                    / query_cost ** cfg["cost_weight"])
        count = min(count, len(pool))
        random_count = min(count, math.ceil(count * self.config["selection"]["random_fraction"]))
        random_part = self.rng.sample(pool, random_count)
        selected = {id(v) for v in random_part}
        ranked = sorted((v for v in pool if id(v) not in selected), key=lambda v: v["acquisition"], reverse=True)
        return random_part + ranked[:count - random_count]

    async def _tail(self, result, policy):
        if result["terminal"] or result["status"] != "truncated":
            return None
        value = await self.evidence.get(result["state"], policy.describe(result["state"]))
        self.teacher_predictions[(result["state"]["policy_version"], state_key(result["state"]))] = {"value": value}
        return value

    async def _query(self, visit, policy, version, *, force_full=False, audit_slots=0, random_audit=None):
        root_prefix = visit["prefix"]
        start = len(root_prefix)
        pool = self.config["model_pool"]
        short = self.config["short_rollout"]
        full = force_full or not short["enabled"] or not self.config["teacher"]["enabled"]
        branches = []
        for model in pool:
            result = None
            if force_full:
                for path in sorted((self.out / "trajectories").glob("*.json")):
                    cached = json.loads(path.read_text())
                    steps = cached.get("all_steps", [])
                    if (cached.get("scope") == "branch" and cached.get("policy_version") == version
                            and cached.get("task_id") == visit["task"]["task_id"] and cached.get("labels_valid")
                            and cached.get("terminal") and len(steps) > start and steps[start]["profile"] == model
                            and [s["call_id"] for s in steps[:start]] == [s["call_id"] for s in root_prefix]):
                        result = cached
                        self.event("cold_branch_reused", trajectory_id=cached["trajectory_id"])
                        break
            if result is None:
                result = await self.runner.rollout(visit["task"], policy, prefix=root_prefix, forced_first=model,
                    horizon=None if full else short["horizon"], scope="branch", policy_version=version)
            branches.append(result)
            if result["status"] == "replay_failed":
                self.event("pair_invalid", reason="prefix_replay_failed_before_other_branch")
                return None, 0
        if any(r["status"] not in {"truncated", "native_done", "EnvironmentDone", "Submitted", "step_limit", "task_budget", "empty_response_limit"}
               and not r["labels_valid"] for r in branches):
            for result in branches:
                await self._real_rows(result, start=start)
            self.event("pair_invalid", statuses=[r["status"] for r in branches])
            return None, 0
        tails = [await self._tail(r, policy) if not full else None for r in branches]
        targets = [result_target(r, start, tail) for r, tail in zip(branches, tails)]
        def uncertain():
            if any(t is None for t in targets):
                return True
            delta = utility(targets[0], self.config) - utility(targets[1], self.config)
            return abs(delta) <= self.config["audit"]["uncertainty_threshold"]
        expanded = short["horizon"] if not full else None
        while not full and not random_audit and short["adaptive"] and uncertain() and expanded < short["max_horizon"]:
            extension = min(short["extension"], short["max_horizon"] - expanded)
            for i, result in enumerate(branches):
                if result["status"] == "truncated":
                    branches[i] = await self.runner.rollout(visit["task"], policy, prefix=result["all_steps"],
                        horizon=extension, scope="branch", policy_version=version)
            expanded += extension
            tails = [await self._tail(r, policy) if r["status"] == "truncated" else None for r in branches]
            targets = [result_target(r, start, tail) for r, tail in zip(branches, tails)]
        before_audit = deepcopy(targets)
        # Missing/insufficient estimates require real continuation, never a failure label.
        must_ground = any(t is None for t in targets)
        if random_audit is None:
            random_audit = self.rng.random() < self.config["audit"]["random_fraction"]
        requested = self.config["audit"]["enabled"] and audit_slots > 0 and (uncertain() or random_audit)
        audited = not full and any(not r["terminal"] for r in branches) and (must_ground or requested)
        if audited:
            for i, result in enumerate(branches):
                if not result["terminal"] and result["status"] == "truncated":
                    branches[i] = await self.runner.rollout(visit["task"], policy, prefix=result["all_steps"],
                        horizon=None, scope="audit", policy_version=version)
            targets = [result_target(r, start) for r in branches]
        pair_id = uuid.uuid4().hex
        # The actual A/B pre-call histories/budgets must agree, not just task+step.
        first = [r["all_steps"][start] if len(r["all_steps"]) > start else None for r in branches]
        if any(s is None for s in first):
            self.event("pair_invalid", reason="no_intervention_call")
            return None, int(audited)
        if state_key(first[0]["before_state"]) != state_key(first[1]["before_state"]):
            raise RuntimeError("A/B interventions did not start from the same public state and budget")
        for result in branches:
            # Only newly intervened suffixes get this outcome; never relabel the
            # inherited prefix with a different counterfactual terminal result.
            await self._real_rows(result, start=start)
        if any(t is None for t in targets):
            self.event("pair_invalid", reason="missing_terminal_or_ready_teacher")
            return None, int(audited)
        delta = utility(targets[0], self.config) - utility(targets[1], self.config)
        std = sum(utility_std(t, self.config) for t in targets)
        actual = all(t["source"] == "real" for t in targets)
        weight = 1. if actual else self.config["optimization"]["pseudo_weight"]
        pair = {"pair_id": pair_id, "state": first[0]["before_state"], "a": pool[0], "b": pool[1],
                "delta": delta, "weight": weight, "source": "real" if actual else "llm_teacher", "policy_version": version,
                "targets": dict(zip(pool, targets)), "rollout_horizon": expanded,
                "audited": audited, "random_audit_selected": random_audit,
                "branch_trajectory_ids": [r["trajectory_id"] for r in branches],
                "intervention_call_ids": [s["call_id"] for s in first],
                "snapshot_id": first[0]["before_state"].get("snapshot_id")}
        assert_training_rows([pair])
        self.pairs[pair_id] = pair
        append_jsonl(self.out / "pairs.jsonl", pair)
        for model, s, target in zip(pool, first, targets):
            self._put_router({"sample_id": s["call_id"], "state": s["before_state"], "model": model,
                "reward": target["reward"], "cost": target["cost"], "weight": 1. if target["source"] == "real" else weight,
                "source": target["source"], "policy_version": version, "pair_id": pair_id,
                "tail_state": branches[pool.index(model)]["state"],
                "actual_suffix_cost": sum(t["cost"] for t in branches[pool.index(model)]["all_steps"][start:])})
        if audited and all(t is not None for t in before_audit):
            predicted = utility(before_audit[0], self.config) - utility(before_audit[1], self.config)
            append_jsonl(self.out / "teacher_audits.jsonl", {"pair_id": pair_id, "policy_version": version,
                "predicted_delta": predicted, "realized_delta": delta, "squared_error": (predicted - delta) ** 2,
                "ranking_agreement": (predicted >= 0) == (delta >= 0), "random_audit": random_audit,
                "endpoint_errors": {m: {"reward_squared_error": (before_audit[i]["reward"]-targets[i]["reward"])**2,
                    "cost_absolute_error": abs(before_audit[i]["cost"]-targets[i]["cost"])}
                    for i,m in enumerate(pool) if before_audit[i]["source"] == "llm_teacher"},
                "warning": "One realized continuation pair is not an exact success-probability oracle"})
        self.event("pair_completed", pair_id=pair_id, source=pair["source"], delta=delta,
                   audited=audited, horizon=expanded, total_cost=self.ledger.cost)
        return pair, int(audited)

    def _train_router(self, version, *, include_cold=False):
        limit = self.config["optimization"]["replay_limit"]
        # The cold initialization may use fixed-model data. Online targets must
        # match this round's continuation policy, including REAL MC returns.
        rows = [r for r in self.router_rows.values() if r["policy_version"] == version
                or (include_cold and r["policy_version"] == -1)][-limit:]
        pairs = [p for p in self.pairs.values() if p["policy_version"] == version][-limit:]
        assert_training_rows(rows + pairs)
        return self.learner.train_router(rows, pairs)

    async def _cold_start(self):
        # Round zero has a known uniform continuation policy. It only trains Router.
        self.evidence.begin_phase()
        version = self.learner.policy_fingerprint(1.)
        policy = FrozenPolicy(self.learner, self.config, version, self.rng, output=self.out, epsilon=1.)
        if self.cold_stage != "uniform":
            tasks = self.rng.sample(self.train, min(self.config["cold_start"]["tasks"], len(self.train)))
            self.cold_query_refs = []
            visits = []
            for task in tasks:
                result = await self.runner.rollout(task, policy, scope="cold_start", policy_version=version)
                await self._real_rows(result)
                visits.extend(self._visits(task, result))
            selected = self.rng.sample(visits, min(self.config["cold_start"]["queries"], len(visits)))
            self.cold_query_refs = [{"trajectory_id": v["trajectory_id"], "prefix_length": len(v["prefix"])} for v in selected]
            self.cold_stage = "uniform"
            self._save(0)
        slots = self.config["audit"]["max_pairs_per_iteration"]
        refs = self.cold_query_refs or []
        random_count = min(len(refs), slots, math.ceil(len(refs)*self.config["audit"]["random_fraction"])) if self.config["audit"]["enabled"] else 0
        for i, ref in enumerate(refs[self.cold_queries_done:], self.cold_queries_done):
            if self.ledger.exhausted:
                break
            result = json.loads((self.out / "trajectories" / (ref["trajectory_id"]+".json")).read_text())
            index = ref["prefix_length"]
            visit = {"task": self.task_map[result["task_id"]], "state": result["all_steps"][index]["before_state"],
                     "prefix": result["all_steps"][:index], "trajectory_id": result["trajectory_id"]}
            _, used = await self._query(visit, policy, version, audit_slots=slots, random_audit=i < random_count)
            slots = max(0, slots-used)
            self.cold_queries_done += 1
            self._save(0)
        report = self._train_router(version)
        self.event("cold_start_finished", policy="uniform", **report)

    async def _evaluate(self, version):
        config = self.config["evaluation"]
        if not config["enabled"] or self.ledger.exhausted:
            return None
        tasks = self.valid if config["max_tasks"] is None else self.valid[:config["max_tasks"]]
        policy_id = self._policy_id(greedy=True)
        policy = self._policy(policy_id, greedy=True)
        progress_path = self.out / "evaluation-progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        planned_ids = [task["task_id"] for task in tasks]
        if (progress.get("policy_version") != policy_id or progress.get("iteration") != version
                or progress.get("task_ids") != planned_ids):
            progress = {"policy_version": policy_id, "iteration": version, "task_ids": planned_ids,
                        "start_cost": self.ledger.cost, "start_reserved_cost": self.ledger.reserved_cost,
                        "start_budget_cost": self.ledger.budget_cost, "results": {}}
        results_by_task = progress["results"]
        for task in tasks:
            if self.ledger.exhausted:
                break
            previous = results_by_task.get(task["task_id"])
            if previous and previous.get("outcome_valid", previous.get("labels_valid", False)):
                continue
            prefix = []
            if previous:
                for step in previous.get("all_steps", []):
                    if step.get("transition") is None:
                        break
                    prefix.append(step)
            # No assessor call, sample ingestion, gradient update or test access.
            try:
                result = await self.runner.rollout(task, policy, prefix=prefix, scope=self.eval_split, policy_version=policy_id)
            except BaseException as exc:
                if hasattr(exc, "rollout_result"):
                    results_by_task[task["task_id"]] = exc.rollout_result
                _atomic(progress_path, progress)
                raise
            results_by_task[task["task_id"]] = result
            _atomic(progress_path, progress)
        results = [results_by_task[t["task_id"]] for t in tasks if t["task_id"] in results_by_task]
        scored = [r for r in results if r.get("outcome_valid", r["labels_valid"])]
        known_cost = self.ledger.cost - progress["start_cost"]
        reserved_cost = self.ledger.reserved_cost - progress["start_reserved_cost"]
        budget_cost = self.ledger.budget_cost - progress["start_budget_cost"]
        # A resumed task replaces its partial result; uncertain failed-attempt
        # fees still belong to this evaluation's ledger interval.
        cost_complete = reserved_cost == 0 and all(r["cost_complete"] for r in results)
        report = {"iteration": version, "policy_version": policy_id, "split": self.eval_split, "planned_tasks": len(tasks),
                  "attempted_tasks": len(results), "scored_tasks": len(scored),
                  "passed": sum(bool(r["success"]) for r in scored),
                  "pass_rate": sum(bool(r["success"]) for r in scored) / len(scored) if scored else None,
                  "mean_reward": sum(r.get("outcome_reward", r["reward"]) for r in scored) / len(scored) if scored else None,
                  "cost": known_cost, "reserved_cost": reserved_cost, "budget_cost": budget_cost,
                  "mean_cost": known_cost / len(results) if results and cost_complete else None,
                  "mean_budget_cost": budget_cost / len(results) if results else None,
                  "currency": self.ledger.currency,
                  "complete": len(scored) == len(tasks), "training_updates": False,
                  "cost_complete": cost_complete, "simulation": self.simulation,
                  "trajectory_ids": [r["trajectory_id"] for r in results]}
        append_jsonl(self.out / "evaluation.jsonl", report)
        if report["complete"]:
            self.last_evaluated = version
        if report["complete"] and (self.best_valid is None or report["pass_rate"] > self.best_valid["pass_rate"]
                or (report["pass_rate"] == self.best_valid["pass_rate"] and report["cost_complete"]
                    and self.best_valid["cost_complete"] and report["cost"] < self.best_valid["cost"])):
            self.best_valid = report
            self.learner.save(self.out / "best_eval.pt", report)
        self.event("validation", **report)
        return report

    async def run(self):
        stop = "iterations_completed"
        if not self.resume:
            self._save(0)
        try:
            self.ledger.check()
            if not self.cold_done:
                if self.config["cold_start"]["enabled"]:
                    await self._cold_start()
                self._save(0, cold_done=True)
            if (self.config["evaluation"]["enabled"] and self.completed_iterations
                    and self.last_evaluated != self.completed_iterations
                    and (self.completed_iterations % self.config["evaluation"]["every_iterations"] == 0
                         or self.completed_iterations == self.config["online"]["iterations"])):
                await self._evaluate(self.completed_iterations)
                self._save(self.completed_iterations)
            for iteration in range(self.completed_iterations + 1, self.config["online"]["iterations"] + 1):
                if self.ledger.exhausted:
                    stop = "global_budget"
                    break
                start_cost = self.ledger.cost
                self.evidence.begin_phase()
                visits = []
                version = self._policy_id()
                policy = self._policy(version)
                self.event("iteration_started", iteration=iteration, policy_version=version, continuation="frozen_epsilon_policy")
                tasks = self.rng.sample(self.train, min(len(self.train), self.config["online"]["tasks_per_iteration"]))
                for task in tasks:
                    if self.ledger.exhausted:
                        break
                    result = await self.runner.rollout(task, policy, scope="collection", policy_version=version)
                    await self._real_rows(result)
                    visits.extend(self._visits(task, result))
                audits = 0
                selected = self._select(visits, self.config["online"]["states_per_iteration"])
                audit_cfg = self.config["audit"]
                # Reserve random audits before uncertainty queries consume their
                # cap. Random selection is independent of teacher confidence.
                random_count = min(len(selected), audit_cfg["max_pairs_per_iteration"],
                    math.ceil(len(selected) * audit_cfg["random_fraction"])) if audit_cfg["enabled"] else 0
                random_indexes = set(self.rng.sample(range(len(selected)), random_count))
                order = sorted(range(len(selected)), key=lambda i: i not in random_indexes)
                for i in order:
                    visit = selected[i]
                    if self.ledger.exhausted:
                        break
                    _, used = await self._query(visit, policy, version,
                        audit_slots=max(0, self.config["audit"]["max_pairs_per_iteration"] - audits),
                        random_audit=i in random_indexes)
                    audits += used
                router = self._train_router(version)
                report = {"iteration": iteration, "policy_version": version, "next_policy_version": self._policy_id(),
                          "visited_states": len(visits), "audited_pairs": audits,
                          "teacher": "fixed_reference_llm_no_updates", "router": router,
                          "iteration_cost": self.ledger.cost - start_cost, "total_cost": self.ledger.cost,
                          "cost_complete": self.ledger.cost_complete,
                          "real_router_samples": sum(r["source"] == "real" for r in self.router_rows.values()),
                          "pseudo_router_samples": sum(r["source"] != "real" for r in self.router_rows.values())}
                append_jsonl(self.out / "training_log.jsonl", report)
                # Commit the updated router before starting paid validation.
                self._save(iteration)
                if iteration % self.config["evaluation"]["every_iterations"] == 0:
                    await self._evaluate(iteration)
                self._save(iteration)
                if iteration % self.config["checkpoint"]["every_iterations"] == 0:
                    self.learner.save(self.out / f"iterations/iteration-{iteration:04d}.pt", report)
                self.event("iteration_finished", **report)
                if not self.ledger.spending_safe:
                    stop = "unknown_cost"
                    break
                if self.ledger.exhausted:
                    stop = "global_budget"
                    break
            if self.completed_iterations and self.last_evaluated != self.completed_iterations:
                await self._evaluate(self.completed_iterations)
                self._save(self.completed_iterations)
        except RecoverableRequestError as exc:
            stop = "invalid_teacher_evidence" if isinstance(exc, InvalidTeacherEvidenceError) else "request_retries_exhausted"
            self._save(self.completed_iterations)
            self.event("stopped", reason=stop, error=str(exc))
        except (BudgetExceeded, UnknownCostError) as exc:
            stop = "unknown_cost" if isinstance(exc, UnknownCostError) else "global_budget"
            self.event("stopped", reason=stop, error=str(exc))
        except BaseException as exc:
            self.event("interrupted", error=f"{type(exc).__name__}: {exc}")
            self._summary("interrupted")
            raise
        return self._summary(stop)

    def _summary(self, stop):
        report = {"schema": "reference-router-run-v2", "stop_reason": stop,
                  "completed_iterations": self.completed_iterations, "cold_start_complete": self.cold_done,
                  "train_tasks": len(self.train), "evaluation_tasks": len(self.valid), "evaluation_split": self.eval_split, "test_tasks_used": len(self.valid) if self.eval_split == "test" else 0,
                  "teacher_llm_profile": self.config["teacher"]["llm"]["profile"],
                  "teacher_llm_enabled": self.config["teacher"]["enabled"],
                  "best_validation": self.best_valid, "simulation": self.simulation,
                  "router_samples": len(self.router_rows), "teacher_updates": 0,
                  "pairs": len(self.pairs), "ledger": self.ledger.report()}
        _atomic(self.out / "summary.json", report)
        (self.out / "SUMMARY.md").write_text(
            f"# Online router training\n\n- Stop: {stop}\n- Completed iterations: {self.completed_iterations}\n"
            f"- Known API cost: {self.ledger.cost:.6f} {self.ledger.currency}\n"
            f"- Cost complete: {self.ledger.cost_complete}\n- Evaluation split: {self.eval_split}\n"
            f"- Synthetic smoke fixture: {self.simulation}\n\n"
            "See evaluation.jsonl for evaluation success/coverage and ledger.json for all scopes.\n")
        return report

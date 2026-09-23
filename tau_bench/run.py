import os
import glob
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy

# attempts within one task are independent; >1 runs them in parallel
ATTEMPT_CONCURRENCY = int(os.getenv("TAU_ATTEMPT_CONCURRENCY", "1"))




def _atomic_write_json(path, payload):
    """Write JSON via a temp file + atomic rename.

    Checkpoints are rewritten in full on every attempt and grow to tens of MB,
    so truncating the real file first means any interruption destroys the whole
    run history. os.replace is atomic on POSIX.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

def find_existing_baseline_folder(config: RunConfig, seed: int) -> Optional[str]:
    """Find a previously-created baseline run folder for this exact config
    (model/agent/env/range/user/seed), ignoring the timestamp suffix, so a
    rerun can resume it instead of starting a fresh folder from scratch.
    """
    prefix = f"{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_seed-{seed}_"
    matches = sorted(glob.glob(os.path.join(config.log_dir, prefix + "*")))
    return matches[-1] if matches else None


def run_baseline(config: RunConfig, existing_folder: Optional[str] = None) -> List[EnvRunResult]:
    random.seed(config.seed)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    if existing_folder and os.path.exists(existing_folder):
        ckpt_folder = existing_folder
        print(f"Resuming existing baseline run folder: {ckpt_folder}")
    else:
        time_str = datetime.now().strftime("%m%d%H%M%S")
        ckpt_folder = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_seed-{config.seed}_{time_str}"
        os.makedirs(ckpt_folder)
    ckpt_path = ckpt_folder + "/transcript.json"

    existing_results: List[EnvRunResult] = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            existing_results = [EnvRunResult(**r) for r in json.load(f)]
    done_task_ids_by_trial: Dict[int, set] = {}
    for r in existing_results:
        done_task_ids_by_trial.setdefault(r.trial, set()).add(r.task_id)

    print(f"running baseline (seed={config.seed})")
    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            all_idxs = config.task_ids
        else:
            all_idxs = list(range(config.start_index, end_index))
        # if config.shuffle:
        #     random.shuffle(idxs)
        # print("idxs:",idxs)

        already_done = done_task_ids_by_trial.get(i, set())
        idxs = [idx for idx in all_idxs if idx not in already_done]
        if already_done:
            print(f"Trial {i}: {len(already_done)} task(s) already completed, {len(idxs)} remaining")
        if not idxs:
            continue

        print(f"N from config {config.best_of_N}")

        def _run(idx: int, N=config.best_of_N) -> EnvRunResult:

            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                # result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                _atomic_write_json(ckpt_path, data + [result.model_dump()])

            # results_intervened.extend

            return result


        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = executor.map(_run, idxs)
            res = list(res)
            results.extend(res)

    results = existing_results + results

    display_metrics(results)

    _atomic_write_json(ckpt_path, [result.model_dump() for result in results])
    print(f"\n📄 Results saved to {ckpt_path}\n")

    return results


def run_baseline_n_times(config: RunConfig, input_path: str) -> List[List[EnvRunResult]]:
    """Re-run an already-completed baseline evaluation additional times, so
    it can be compared against an intervention run.

    `input_path` should point at the folder of the baseline run that was
    already completed with seed=config.seed (the same seed that will be/was
    used for the intervention run). That run is left untouched. This
    function performs config.num_trials - 1 additional full baseline runs,
    each with its own distinct seed, each written to its own new folder
    under config.log_dir. Each individual seed run is exactly 1 trial —
    config.num_trials only controls how many distinct seeds are produced
    here, not how many times each seed's task set is repeated.

    Reruns of this function skip seeds whose folder already has every task
    completed, and resume (only running the missing task IDs) for seeds
    whose folder exists but is incomplete.
    """
    assert config.num_trials >= 1, "num_trials must be >= 1"
    assert os.path.exists(input_path), f"input_path does not exist: {input_path}"

    base_seed = config.seed
    new_seeds = [base_seed + i for i in range(1, config.num_trials)]

    print(f"Already-completed baseline run (seed={base_seed}): {input_path}")
    print(f"Running {len(new_seeds)} additional baseline run(s) with seeds: {new_seeds}")

    all_results: List[List[EnvRunResult]] = []
    for seed in new_seeds:
        run_config = config.model_copy(update={"seed": seed, "run_intervention": False, "num_trials": 1})
        existing_folder = find_existing_baseline_folder(run_config, seed)
        if existing_folder:
            print(f"Seed {seed}: found existing folder {existing_folder}, resuming any incomplete tasks")
        results = run_baseline(run_config, existing_folder=existing_folder)
        all_results.append(results)

    return all_results


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot", "react-intervened", "react-reflexion"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")

    if (not config.run_intervention):
        return run_baseline(config)

    #Run interventions
    baseline_path = config.baseline_path
    intervention_folder = baseline_path + f"/intervened-by_{config.intervention_model}_{time_str}"
    os.makedirs(intervention_folder, exist_ok=True)

    print(f"saving to: {intervention_folder}")
    ckpt_path_intervened = intervention_folder + "/intervened-transcripts.json"
    agent_conv_history_path = intervention_folder + '/agent_conversation_history.json'

    examples = []
    with open(baseline_path + "/transcript.json","r", encoding="utf-8") as json_file:
        examples = json.load(json_file)

    if config.task_ids and len(config.task_ids) > 0:
        examples = [e for e in examples if e["task_id"] in set(config.task_ids)]

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Running interventions by: {config.intervention_model}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    results_intervened: List[EnvRunResult] = []
    total = []
    improved_count = []
    intervened_succesful_count = []
    agent_conversation_histories = []
    lock = multiprocessing.Lock()
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        # if config.shuffle:
        #     random.shuffle(idxs)

        print(f"N from config {config.best_of_N}")

        def _run(baseline_example: dict, N=config.best_of_N) -> EnvRunResult:
            idx = baseline_example["task_id"]

            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )
            result = EnvRunResult(
                task_id=idx,
                reward=baseline_example["reward"],
                info=baseline_example["info"],
                traj=baseline_example["traj"],
                trial=i,
            )

            print(f"Running interventions on {idx}")

            try:
                if result.reward == 1.0:
                    total.append(True)
                    intervened_succesful_count.append(1.0)
                    agent_conversation_histories.append({"task_id":idx,"traj":"None. Reward is already 1."})
                    result_intervened = EnvRunResult(
                        intervened_message = "no intervention was needed. already passed",
                        intervened_index = str(-1),
                        improved = 0,
                        task_id=idx,
                        reward=result.reward,
                        info=result.info,
                        traj=result.traj,
                        trial=i,
                    )
                    print(f"already passed so no intervention needed. task_id: {idx}")
                    with lock:
                        data = []
                        if os.path.exists(ckpt_path_intervened):
                            with open(ckpt_path_intervened, "r") as f:
                                data = json.load(f)
                        _atomic_write_json(ckpt_path_intervened, data + [result_intervened.model_dump()])
                        # continue
                else:

                    #get intervention possibilites and conv history of intervening agent
                    print(f"determining interventions right now {idx}")
                    answer_list, intervening_agent_conversation = agent.run_intervention(
                        env=isolated_env,
                        task_index=idx,
                        result=result,
                        N=N
                    ) 
                    print(f"done {idx}")
                    does_improve = False
                    passed_intervened = False
                    best_intervened_score = 0
                    if (answer_list == None or answer_list[0] == False):
                        # total.append(True)
                        intervened_succesful_count.append(0.0)
                        agent_conversation_histories.append({"task_id":idx,"traj":intervening_agent_conversation})
                        result_intervened = EnvRunResult(
                            intervened_message = "no intervention was done",
                            failure_brief = "no intervention was done",
                            failure_index = str(-1),
                            intervened_index = str(-1),
                            improved = 0,
                            task_id=idx,
                            reward=result.reward,
                            success_prev = result.reward,
                            success_after = None,
                            info=result.info,
                            traj=result.traj,
                            trial=i,
                        )
                        with lock:
                                data = []
                                if os.path.exists(ckpt_path_intervened):
                                    with open(ckpt_path_intervened, "r") as f:
                                        data = json.load(f)
                                _atomic_write_json(ckpt_path_intervened, data + [result_intervened.model_dump()])
                        # if (answer_list[0] != False):
                        total.append(True)
                        if (does_improve == True):
                            improved_count.append(True)
                        # if (passed_intervened == True):
                        intervened_succesful_count.append(best_intervened_score)

                        
                    else:
                        #add conv history to all conv histories
                        agent_conversation_histories.append({"task_id":idx,"traj":intervening_agent_conversation})

                        transcript = result.model_dump()
                        possible_new_trajectories = []
                        

                        def add_intervention(trajectory, intervention_text, intervention_id):
                            try:
                                if type(intervention_id) == int:
                                    idx_intervention =  intervention_id  
                                else:
                                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                                    idx_intervention = int(intervention_id[idx_b+1:])
                            except Exception as e:
                                print(f"first converting to int idx: {idx} error: {e}.")
                                idx_intervention = 1

                            new_trajectory = trajectory[:min(idx_intervention+1, len(trajectory) - 1)]
                            if (len(trajectory) - 1 < idx_intervention+1):
                                print(f"was out of range task id: {idx}")
                            new_trajectory.append(
                                {
                                    "role":"system",
                                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                                }
                            )

                            return new_trajectory
                        

                        # print("possible interventions:", answer_list)

                        
                        for best_of_n_iterator, intervention in enumerate(answer_list):
                            try:

                                intervention_id = intervention["id"]
                                if type(intervention_id) == int:
                                    idx_intervention =  intervention_id  
                                else:
                                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                                    if idx_b == -1:
                                        idx_b = -1 if (intervention_id.rfind("A") == -1) else intervention_id.rfind("A")
                                    idx_intervention = int(intervention_id[idx_b+1:])

                            except Exception as e:
                                print(f"converting to int idx: {idx} error: {e}.")
                                idx_intervention = 1

                            answer_list[best_of_n_iterator]["id"] = idx_intervention

                        for best_of_n_iterator, intervention in enumerate(answer_list):
                            try:

                                failure_id = intervention["failure_id"]
                                if type(failure_id) == int:
                                    idx_failure =  failure_id  
                                else:
                                    idx_b = -1 if (failure_id.rfind("B") == -1) else failure_id.rfind("B")
                                    if idx_b == -1:
                                        idx_b = -1 if (failure_id.rfind("A") == -1) else failure_id.rfind("A")
                                    idx_failure = int(failure_id[idx_b+1:])

                            except Exception as e:
                                print(f"failure id; converting to int idx: {idx} error: {e}.")
                                idx_failure = 1

                            answer_list[best_of_n_iterator]["failure_id"] = idx_failure

                        sorted_answer_list = sorted(answer_list, key=lambda x: x['id'])
                        
                        # print("sorted possible interventions:", sorted_answer_list)

                        #loop through all intervention possibilites
                        # for best_of_n_iterator, intervention in enumerate([sorted_answer_list[0], sorted_answer_list[-1]]):
                        def _attempt(item):
                            """Run one intervention attempt. Attempts are independent, so they run
                            in parallel; each needs its own env because Env carries rollout state."""
                            best_of_n_iterator, intervention = item
                            print(f"trying out task id={idx}, intervention {best_of_n_iterator}")
                            failure_brief = intervention["failure_brief"]
                            failure_id = intervention["failure_id"]
                            intervention_text = intervention["intervention_text"]
                            intervention_id = intervention["id"]
                            new_intervened_trajectory = add_intervention(transcript["traj"], intervention_text, intervention_id)
                            attempt_env = get_env(
                                config.env,
                                user_strategy=config.user_strategy,
                                user_model=config.user_model,
                                task_split=config.task_split,
                                user_provider=config.user_model_provider,
                                task_index=idx,
                            )
                            res_intervened = agent.solve_with_intervention(
                                env=attempt_env, task_index=idx, messages=new_intervened_trajectory
                            )
                            result_intervened = EnvRunResult(
                                failure_brief=failure_brief,
                                failure_index=str(failure_id),
                                intervened_message=intervention_text,
                                intervened_first_or_last=str(best_of_n_iterator),
                                intervened_index=str(intervention_id),
                                improved=(result.reward == 0 and res_intervened.reward != 0),
                                success_prev=result.reward,
                                success_after=res_intervened.reward,
                                task_id=idx,
                                reward=res_intervened.reward,
                                info=res_intervened.info,
                                traj=res_intervened.messages,
                                trial=i,
                            )
                            if result.reward == 0 and result_intervened.reward != 0:
                                print(f"***IMPROVED*** task_id={idx} at location={intervention_id}")
                            with lock:
                                data = []
                                if os.path.exists(ckpt_path_intervened):
                                    with open(ckpt_path_intervened, "r") as f:
                                        data = json.load(f)
                                _atomic_write_json(ckpt_path_intervened, data + [result_intervened.model_dump()])
                            return result_intervened

                        attempt_workers = max(1, min(ATTEMPT_CONCURRENCY, len(sorted_answer_list)))
                        with ThreadPoolExecutor(max_workers=attempt_workers) as attempt_pool:
                            attempt_results = list(attempt_pool.map(_attempt, list(enumerate(sorted_answer_list))))
                        for r_att in attempt_results:
                            if result.reward == 0 and r_att.reward != 0:
                                does_improve = True
                            if r_att.reward != 0:
                                passed_intervened = True
                                best_intervened_score = max(r_att.reward, best_intervened_score)
                        if attempt_results:
                            result_intervened = max(attempt_results, key=lambda r: r.reward)

                        # if (answer_list[0] != False):
                        total.append(True)
                        if (does_improve == True):
                            improved_count.append(True)
                        # if (passed_intervened == True):
                        intervened_succesful_count.append(best_intervened_score)

            except Exception as e:
                result_intervened = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
                print(f"task id: {idx}. error: {e}.")
                with lock:
                    data = []
                    if os.path.exists(ckpt_path_intervened):
                        with open(ckpt_path_intervened, "r") as f:
                            data = json.load(f)
                    _atomic_write_json(ckpt_path_intervened, data + [result_intervened.model_dump()])

            
            try:
                print(
                    "✅" if result_intervened.reward == 1 else "❌",
                    f"intervened task_id={idx}",
                    # result_intervened.info,
                )
            except:
                pass
            
            return result_intervened

        
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = executor.map(_run, examples)
            res = list(res)
            results_intervened.extend(res)

    display_metrics(results_intervened)

    if (config.run_intervention):
        for leftover in range (max(0, (end_index - config.start_index) - len(intervened_succesful_count))):
            intervened_succesful_count.append(0)

        average = sum(intervened_succesful_count) / len(total)
        print("Average reward (intervened):", average)
        print("total samples:", len(total))
        print("# of samples with intervention with score > 0:", len(intervened_succesful_count))
        print("# of samples improved from a score of 0 with intervention:", len(improved_count))
        # print("percent of samples passed with intervention:", len(intervened_succesful_count) / len(total))
        print("percent of missed samples improved with intervention:", len(improved_count) / len(intervened_succesful_count))
        print("scores for interventions:", intervened_succesful_count)


        with open(agent_conv_history_path, 'w') as json_file:
            json.dump(agent_conversation_histories, json_file, indent=4)
            print(f"Intervention data successfully written to '{intervention_folder}'")





    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    
    elif config.agent_strategy == "react-intervened":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent_intervened import ChatReActAgentIntervened

        return ChatReActAgentIntervened(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
            intervenor_temperature=config.intervenor_temperature,
            intervention_model=config.intervention_model,
            intervention_provider=config.intervention_model_provider or "openai",
            search_model=config.search_model or config.intervention_model,
            search_provider=config.search_model_provider
            or (config.intervention_model_provider or "openai"),
        )
    
    elif config.agent_strategy == "react-reflexion":
        from tau_bench.agents.chat_react_agent_reflexion import ChatReActAgentReflexion

        return ChatReActAgentReflexion(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )

    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")

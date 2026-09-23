# Copyright Sierra

import copy
import json
from litellm import completion
from litellm.exceptions import BadRequestError
from tau_bench.llm_utils import completion_with_backoff
from docent import Docent

from openai import OpenAI
import openai
import os 
api_key = os.getenv('OPENAI_API_KEY')
# every LLM call in this module goes through litellm; this client is kept only for
# backwards compatibility and must not make importing the module require an OpenAI key
client = OpenAI(api_key=api_key) if api_key else None


#docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message
from docent.samples import get_inspect_fpath
from pydantic_core import to_jsonable_python
from typing import Any
from docent.data_models import BaseAgentRunMetadata
from pydantic import Field
from docent.data_models.chat import SystemMessage, UserMessage, AssistantMessage, ToolMessage, ContentReasoning
import re
from tau_bench.agents.intervening_prompts import questioning_agent_prompt, SEARCH_PROMPT, SINGLE_RUN_CITE_INSTRUCTION, questioning_agent_prompt_working_backwards, questioning_agent_prompt_working_backwards_react
import pprint
from docent.data_models.chat import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ContentText,
    ContentReasoning,
    ToolCall,
)

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.types import (
    Action, 
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple

class CustomTauAgentRunMetadata(BaseAgentRunMetadata):
    task_id: str = Field(
        description="The ID of the 'benchmark' or 'set of evals' that the transcript belongs to"
    )

    sample_id: str = Field(
        description="The specific task inside of the `task_id` benchmark that the transcript was run on"
    )
    epoch_id: int = Field(
        description="Each `sample_id` should be run multiple times due to stochasticity; `epoch_id` is the integer index of a specific run."
    )

    model: str = Field(description="The model that was used to generate the transcript")

    scoring_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the scoring process"
    )

    additional_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the transcript"
    )



class ChatReActAgentIntervened(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 1.0,
        intervenor_temperature: float = 0.2,
        intervention_model: str = "gpt-4o-mini-2024-07-18",
        intervention_provider: str = "openai",
        search_model: Optional[str] = None,
        search_provider: Optional[str] = None,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.intervenor_temperature = intervenor_temperature
        self.intervention_model = intervention_model
        self.intervention_provider = intervention_provider
        # model backing the docent_query_tool (transcript search); held separate
        # from the intervenor so tool quality does not confound intervenor quality
        self.search_model = search_model or intervention_model
        self.search_provider = search_provider or intervention_provider
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

    @staticmethod
    def _strict_order_compat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rewrite a resumed trajectory for providers (e.g. Mistral) that reject
        conversations not ending in a user/tool message.

        Mid-conversation system messages (the inserted [*INTERVENTION*...] text)
        become user messages with identical content, and a trailing assistant
        message is dropped so the model regenerates that step instead of
        continuing after it.
        """
        msgs = []
        for i, m in enumerate(messages):
            content = m.get("content")
            if not content:
                tc = m.get("tool_calls") or m.get("function_call")
                content = json.dumps(tc, default=str) if tc else "(empty)"
            clean = {"role": m.get("role", "user"), "content": content}
            if i > 0 and clean["role"] == "system":
                clean["role"] = "user"
            msgs.append(clean)
        while msgs and msgs[-1].get("role") == "assistant":
            msgs.pop()
        return msgs


    @staticmethod
    def _portable_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rewrite the intervenor conversation without OpenAI-specific tool plumbing.

        Non-OpenAI endpoints (Llama/Mistral/DeepSeek/Kimi on Azure AI) reject an
        assistant message carrying `tool_calls` plus a `role: "tool"` reply when no
        tool schema was declared. The search result is folded into a plain user
        message instead, which preserves what the intervenor sees.
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "tool":
                out.append({"role": "user", "content": "[SEARCH TOOL RESULT]\n" + (m.get("content") or "")})
                continue
            m.pop("tool_calls", None)
            m.pop("tool_call_id", None)
            if not m.get("content"):
                m["content"] = "(empty)"
            out.append(m)
        return out

    def _intervenor_completion(self, conversation_history: List[Dict[str, Any]]):
        """Intervenor call with a provider-portable retry."""
        try:
            return completion_with_backoff(
                model=self.intervention_model,
                custom_llm_provider=self.intervention_provider,
                num_retries=5,
                messages=conversation_history,
                max_completion_tokens=4096,
                temperature=self.intervenor_temperature,
            )
        except BadRequestError:
            return completion_with_backoff(
                model=self.intervention_model,
                custom_llm_provider=self.intervention_provider,
                num_retries=5,
                messages=self._portable_history(conversation_history),
                max_tokens=4096,
                temperature=self.intervenor_temperature,
            )

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        try:
            res = completion_with_backoff(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
            )
        except BadRequestError:
            # retry once with a sanitized trajectory; harmless if it was not
            # a message-format rejection (the same error just re-raises)
            res = completion_with_backoff(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=self._strict_order_compat(messages),
                temperature=self.temperature,
            )
        message = res.choices[0].message
        # print("***message:",message)
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }

        if "name" not in action_parsed or "arguments" not in action_parsed:
            print(f"Failed message output format: {message}")
            # with open("failed_messages.txt", "a", encoding="utf-8") as f:
            #     f.write(str(message))
            #     f.write("\n" + "-"*80 + "\n")
            return None, None, None
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        return message.model_dump(), action, (res._hidden_params.get("response_cost") or 0.0)

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages_og: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        # print("here")
        while True:
            messages = copy.deepcopy(messages_og)
            total_cost = 0.0
            info = {}
            restart = False
            for _ in range(max_num_steps):
                message, action, cost = self.generate_next_step(messages)
                if message == None or action == None or cost == None:
                    print(f"restarting it, {task_index}")
                    restart = True
                    break
                response = env.step(action)
                obs = response.observation
                reward = response.reward
                info = {**info, **response.info.model_dump()}
                if action.name != RESPOND_ACTION_NAME:
                    obs = "API output: " + obs
                # print(obs)
                messages.extend(
                    [
                        message,
                        {"role": "user", "content": obs},
                    ]
                )
                total_cost += cost
                if response.done:
                    break
            if restart:
                continue
            return SolveResult(
                messages=messages,
                reward=reward,
                info=info,
            )


    def add_intervention(trajectory, intervention_text, intervention_id):
        if type(intervention_id) == int:
            idx_intervention =  intervention_id  
        else:
            idx_b = -1 if (intervention_id.find("B") == -1) else intervention_id.find("B")
            idx_intervention = int(intervention_id[idx_b+1:])

        new_trajectory = trajectory[:idx_intervention+1]
        new_trajectory.append(
            {
                "role":"system",
                "content": "[*INTERVENTION*: " +intervention_text + "]"
            }
        )

        return new_trajectory

    def run_intervention(
            self, N, result: EnvRunResult, env: Env, task_index: Optional[int] = None
    ):
        
        def load_TAU_Reasoning_inspect_log(log) -> list[AgentRun]:
            agent_runs: list[AgentRun] = []
            for sample in log:
                scores: dict[str, int | float | bool] = {}
                scores["correct"] =  (sample["reward"] == 1.0)

                metadata = CustomTauAgentRunMetadata(
                        task_id="airline",
                        sample_id=str(sample["task_id"]),
                        epoch_id=int(sample["trial"]),
                        model="gpt-4o-mini-2024-07-18",
                        scores=scores,
                        additional_metadata=None,
                        scoring_metadata=None,
                    )
                

                messages = []

                for idx, message in enumerate(sample["traj"]):
                    if message["role"] == "tool":
                        messages.append(ToolMessage(id = str(idx),content=message["content"], tool_call_id=message.get("tool_call_id", ""), function=message.get("name", "")))

                    elif message["role"] == "assistant":
                        contentstr = message["content"]
                        if contentstr == None:
                            contentstr = ""
                        if message.get("tool_calls"):
                            contentstr += json.dumps(message["tool_calls"], indent=2)
                        if message.get("function_call"):
                            contentstr += str(message["function_call"])
                        if message.get("annotations"):
                            contentstr += message["annotations"]

                        messages.append(AssistantMessage(id = str(idx),content=contentstr))

                    else:
                        messages.append(UserMessage(id = str(idx),content=message["content"]))


                agent_runs.append(
                AgentRun(
                    transcripts={
                        "default": Transcript(
                            messages=messages
                        )
                    },
                    metadata=metadata,
                )
                )

            return agent_runs

        def execute_search(text, query, model):

            response = completion_with_backoff(
                model=self.search_model,
                custom_llm_provider=self.search_provider,
                num_retries=5,
                messages=[{"role": "user","content":SEARCH_PROMPT.format(text=text, search_query=query, SINGLE_RUN_CITE_INSTRUCTION=SINGLE_RUN_CITE_INSTRUCTION)}],
                max_completion_tokens=4096,
                temperature = self.intervenor_temperature
            )



            return (response.choices[0].message.content or "").strip()

        def load_TAU_reAct_extra_data(transcript):
            return transcript["traj"][0], transcript["info"]["reward_info"], transcript["info"]["task"]

        def add_intervention(trajectory, intervention_text, intervention_id):
            idx_b = -1 if (intervention_id.find("B") == -1) else intervention_id.find("B")
            idx_intervention = int(intervention_id[idx_b+1:])

            new_trajectory = trajectory[:idx_intervention+1]
            new_trajectory.append(
                {
                    "role":"system",
                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                }
            )

            return new_trajectory



        transcript = result.model_dump()
        try: 
            val = (transcript["traj"][0])
        except Exception as e:
            print(f"mcdonalds here error: {e},", transcript["traj"])
        # print("transcript sdfasdf:",transcript["traj"])
        specification, metadata, user_task = load_TAU_reAct_extra_data(transcript)
        agent_run_docent = load_TAU_Reasoning_inspect_log([transcript])

        transcript_length = len(transcript["traj"])

        print("N:", N)


        conversation_history = []
        conversation_history.append({
            "role": "system",
            "content": questioning_agent_prompt_working_backwards.format(specification=specification, ref_metadata=user_task, N=N, transcript_length = transcript_length),
        })   
        
        if self.intervention_provider not in ("openai", "azure"):
            # open-weight endpoints continue a system-only prompt instead of acting
            # on it; an explicit user turn is required to start the intervenor loop
            conversation_history.append({"role": "user", "content": "Begin."})

        turns = 0

        while turns < 30:
            try:
                turns+=1
                # print("turn:", turns)
                response = self._intervenor_completion(conversation_history)

                reply = (response.choices[0].message.content or "").strip()
                if not reply:
                    # reasoning models return the tagged output on reasoning_content
                    reply = (getattr(response.choices[0].message, "reasoning_content", None) or "").strip()
                match = re.search(r'<query>(.*?)</query>', reply, re.DOTALL)

                conversation_history.append({
                    "role": "assistant",
                    "content": reply,
                })



                if match:
                    query_text = match.group(1).strip()
                    conversation_history[-1]["tool_calls"] = [{"function": { "arguments": query_text, "name": "querying_tool"      }, "id": "12345","type": "function"}]
                    tool_response = execute_search(agent_run_docent[0].transcripts["default"].to_str(), query_text, self.model).strip()

                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": "12345",
                        "content": tool_response
                    })
                    continue



                match = re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL)

                if match:
                    answer_text = match.group(1).strip()
                    try:
                        

                        answer_list = json.loads(answer_text)
                        # print("answer:", answer_list)

                        possible_new_trajectories = []
                        # for intervention in answer_list:
                        #     intervention_text = intervention["intervention_text"]
                        #     intervention_id = intervention["id"]
                            
                            # possible_new_trajectories.append(add_intervention(transcript["traj"], intervention_text, intervention_id))

                    
                        # pprint.pprint(conversation_history)
                        
                    

                        return answer_list, conversation_history

                    except json.JSONDecodeError:
                        print("Error decoding JSON.")
                    break


                else:
                    print("model did not call query tool or generate intervention")
                    break
            except Exception as e:
                print(f"error: {e}.")
                break

        

        print("no changes with intervention, error must have happened")
        return [False], conversation_history
        

    def run_intervention_react(
        self, N, result: EnvRunResult, env: Env, task_index: Optional[int] = None
    ):
    
        def load_TAU_Reasoning_inspect_log(log) -> list[AgentRun]:
            agent_runs: list[AgentRun] = []
            for sample in log:
                scores: dict[str, int | float | bool] = {}
                scores["correct"] =  (sample["reward"] == 1.0)

                metadata = CustomTauAgentRunMetadata(
                        task_id="airline",
                        sample_id=str(sample["task_id"]),
                        epoch_id=int(sample["trial"]),
                        model="gpt-4o-mini-2024-07-18",
                        scores=scores,
                        additional_metadata=None,
                        scoring_metadata=None,
                    )
                

                messages = []

                for idx, message in enumerate(sample["traj"]):
                    if message["role"] == "tool":
                        messages.append(ToolMessage(id = str(idx),content=message["content"], tool_call_id=message.get("tool_call_id", ""), function=message.get("name", "")))

                    elif message["role"] == "assistant":
                        contentstr = message["content"]
                        if contentstr == None:
                            contentstr = ""
                        if message.get("tool_calls"):
                            contentstr += json.dumps(message["tool_calls"], indent=2)
                        if message.get("function_call"):
                            contentstr += str(message["function_call"])
                        if message.get("annotations"):
                            contentstr += message["annotations"]

                        messages.append(AssistantMessage(id = str(idx),content=contentstr))

                    else:
                        messages.append(UserMessage(id = str(idx),content=message["content"]))


                agent_runs.append(
                AgentRun(
                    transcripts={
                        "default": Transcript(
                            messages=messages
                        )
                    },
                    metadata=metadata,
                )
                )

            return agent_runs

        def execute_search(text, query, model):

            response = completion_with_backoff(
                model=self.search_model,
                custom_llm_provider=self.search_provider,
                num_retries=5,
                messages=[{"role": "user","content":SEARCH_PROMPT.format(text=text, search_query=query, SINGLE_RUN_CITE_INSTRUCTION=SINGLE_RUN_CITE_INSTRUCTION)}],
                max_completion_tokens=4096,
                temperature = self.intervenor_temperature
            )



            return (response.choices[0].message.content or "").strip()

        def load_TAU_reAct_extra_data(transcript):
            return transcript["traj"][0], transcript["info"]["reward_info"], transcript["info"]["task"]

        def add_intervention(trajectory, intervention_text, intervention_id):
            idx_b = -1 if (intervention_id.find("B") == -1) else intervention_id.find("B")
            idx_intervention = int(intervention_id[idx_b+1:])

            new_trajectory = trajectory[:idx_intervention+1]
            new_trajectory.append(
                {
                    "role":"system",
                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                }
            )

            return new_trajectory

    

        transcript = result.model_dump()
        specification, metadata, user_task = load_TAU_reAct_extra_data(transcript)
        agent_run_docent = load_TAU_Reasoning_inspect_log([transcript])

        print("N:", N)


        conversation_history = []
        conversation_history.append({
            "role": "system",
            "content": questioning_agent_prompt_working_backwards_react.format(specification=specification, ref_metadata=user_task,reward_info=metadata, N=N),
        })   
        
        if self.intervention_provider not in ("openai", "azure"):
            conversation_history.append({"role": "user", "content": "Begin."})

        turns = 0
        while turns < 30:
            # break
            turns += 1
            print("turn:", turns)
            response = self._intervenor_completion(conversation_history)

            
            reply = (response.choices[0].message.content or "").strip()
            if not reply:
                # reasoning models return the tagged output on reasoning_content
                reply = (getattr(response.choices[0].message, "reasoning_content", None) or "").strip()

            # print(reply)

            action_str = reply.split("Action:")[-1].strip()


            try:
                action_parsed = json.loads(action_str)
            except json.JSONDecodeError:
                # this is a hack
                action_parsed = {
                    "name": RESPOND_ACTION_NAME,
                    "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
                }

            assert "name" in action_parsed
            assert "arguments" in action_parsed
            # action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

            # print(action_parsed)
        


            match = re.search(r'<query>(.*?)</query>', reply, re.DOTALL)

            conversation_history.append({
                "role": "assistant",
                "content": reply,
            })



            if action_parsed["name"] == "docent_query_tool":
                query_text = action_parsed["arguments"]["query"].strip()


                conversation_history[-1]["tool_calls"] = [{"function": { "arguments": query_text, "name": "docent_querying_tool"      }, "id": "12345","type": "function"}]
                tool_response = execute_search(agent_run_docent[0].transcripts["default"].to_str(), query_text, "gpt-4o-mini-2024-07-18").strip()

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": "12345",
                    "content": tool_response
                })
                # continue
            

            else:
            
                content_txt = action_parsed["arguments"]["content"]

                match = re.search(r'<answer>(.*?)</answer>', content_txt, re.DOTALL)

                if match:
                    answer_text = match.group(1).strip()
                    try:


                        answer_list = json.loads(answer_text)
                        print("answer:", answer_list)

                        # possible_new_trajectories = []
                        # for intervention in answer_list:
                        #     intervention_text = intervention["intervention_text"]
                        #     intervention_id = intervention["id"]

                        #     possible_new_trajectories.append(add_intervention(transcript["traj"], intervention_text, intervention_id))


                        # pprint.pprint(conversation_history)
                        return answer_list, conversation_history

                    except json.JSONDecodeError:
                        print("Error decoding JSON.") 
                        break


                else:
                    print("model did not call query tool or generate intervention")
                    break

        return None ,conversation_history





    def solve_with_intervention(
        self, messages, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        while True:
            messages_new = copy.deepcopy(messages)
            reward = 0.0
            total_cost = 0.0
            info = {}
            restart = False


            for _ in range(max_num_steps):
                message, action, cost = self.generate_next_step(messages_new)
                if (message == None and action == None and cost == None):
                    print(f"restarting intervened task {task_index}")
                    restart = True
                    break
                response = env.step(action)
                obs = response.observation
                reward = response.reward
                info = {**info, **response.info.model_dump()}
                if action.name != RESPOND_ACTION_NAME:
                    obs = "API output: " + obs
                messages_new.extend(
                    [
                        message,
                        {"role": "user", "content": obs},
                    ]
                )
                total_cost += cost
                if response.done:
                    break
            if (restart):
                continue
            return SolveResult(
                messages=messages_new,
                reward=reward,
                info=info,
            )
    



REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""

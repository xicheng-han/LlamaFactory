# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit LlamaFactory's prompt pipeline against Transformers before training."""

from __future__ import annotations

import copy
import inspect
import json
import logging as std_logging
import os
import signal
import tempfile
import threading
import time
import traceback
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..extras import logging


CUTOFF_LEN = 4096
IGNORE_INDEX = -100
LOSS_EXCLUDED = -1
LOSS_INCLUDED = 0
REPORT_NAME = "template_consistency.json"
SAMPLE_SIZE = 100
SAMPLE_TIMEOUT = 5.0
THINK = "<think>"
END_THINK = "</think>"

logger = logging.get_logger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@contextmanager
def _suppress_runtime_output() -> Iterator[None]:
    previous_logging_level = std_logging.root.manager.disable
    std_logging.disable(std_logging.CRITICAL)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        std_logging.disable(previous_logging_level)


def _case(
    case_id: str,
    enable_thinking: bool,
    system: str,
    conversations: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "id": case_id,
        "enable_thinking": enable_thinking,
        "system": system,
        "conversations": conversations,
    }
    if tools:
        result["tools"] = tools
    return result


CASES = [
    _case(
        "single_turn_without_system_with_thinking",
        True,
        "",
        [
            {"from": "human", "value": "What is 17 multiplied by 24?"},
            {"from": "gpt", "value": f"{THINK}\nBreak 24 into 20 and 4.\n{END_THINK}\n\n408"},
        ],
    ),
    _case(
        "single_turn_with_system_and_thinking",
        True,
        "You are a helpful math assistant. Give a concise final answer.",
        [
            {"from": "human", "value": "What is 17 multiplied by 24?"},
            {"from": "gpt", "value": f"{THINK}\nBreak 24 into 20 and 4.\n{END_THINK}\n\n408"},
        ],
    ),
    _case(
        "multi_turn_with_system_and_thinking",
        True,
        "You are a helpful math assistant. Give a concise final answer.",
        [
            {
                "from": "human",
                "value": "A box contains 12 red markers and 8 blue markers. How many markers are there?",
            },
            {"from": "gpt", "value": f"{THINK}\n12 + 8 = 20.\n{END_THINK}\n\nThere are 20 markers."},
            {"from": "human", "value": "If 5 markers are removed, how many remain?"},
            {"from": "gpt", "value": f"{THINK}\n20 - 5 = 15.\n{END_THINK}\n\n15 markers remain."},
        ],
    ),
    _case(
        "tool_call_multi_turn_with_system_and_thinking",
        True,
        "You are a helpful weather assistant. Use the available tools when needed.",
        [
            {"from": "human", "value": "What is the current temperature in Paris?"},
            {
                "from": "function_call",
                "value": (
                    f"{THINK}\nI should use the weather tool for the current temperature in Paris.\n"
                    f'{END_THINK}\n\n{{"name":"get_current_temperature","arguments":{{"city":"Paris"}}}}'
                ),
            },
            {"from": "observation", "value": '{"temperature_celsius":21}'},
            {
                "from": "gpt",
                "value": f"{THINK}\nThe tool returned 21 degrees Celsius.\n{END_THINK}\n\n21 degrees Celsius.",
            },
        ],
        tools=TOOLS,
    ),
    _case(
        "single_turn_without_system",
        False,
        "",
        [{"from": "human", "value": "What is 17 multiplied by 24?"}, {"from": "gpt", "value": "408"}],
    ),
    _case(
        "single_turn_with_system",
        False,
        "You are a helpful math assistant. Give a concise final answer.",
        [{"from": "human", "value": "What is 17 multiplied by 24?"}, {"from": "gpt", "value": "408"}],
    ),
    _case(
        "multi_turn_with_system",
        False,
        "You are a helpful math assistant. Give a concise final answer.",
        [
            {
                "from": "human",
                "value": "A box contains 12 red markers and 8 blue markers. How many markers are there?",
            },
            {"from": "gpt", "value": "There are 20 markers."},
            {"from": "human", "value": "If 5 markers are removed, how many remain?"},
            {"from": "gpt", "value": "15 markers remain."},
        ],
    ),
    _case(
        "tool_call_multi_turn_with_system",
        False,
        "You are a helpful weather assistant. Use the available tools when needed.",
        [
            {"from": "human", "value": "What is the current temperature in Paris?"},
            {"from": "function_call", "value": '{"name":"get_current_temperature","arguments":{"city":"Paris"}}'},
            {"from": "observation", "value": '{"temperature_celsius":21}'},
            {"from": "gpt", "value": "The current temperature in Paris is 21 degrees Celsius."},
        ],
        tools=TOOLS,
    ),
]


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _compare_ids(left: list[int], right: list[int]) -> str:
    return "same" if left == right else "different"


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _loss_participation(tokenizer: Any, input_ids: list[int], labels: list[int]) -> list[dict[int, str]]:
    """Group adjacent tokens by whether they participate in loss."""
    if len(input_ids) != len(labels):
        raise ValueError(f"input_ids/labels length mismatch: {len(input_ids)} != {len(labels)}")
    if not input_ids:
        return []

    result: list[dict[int, str]] = []
    start = 0
    current = LOSS_EXCLUDED if labels[0] == IGNORE_INDEX else LOSS_INCLUDED
    for index in range(1, len(labels)):
        value = LOSS_EXCLUDED if labels[index] == IGNORE_INDEX else LOSS_INCLUDED
        if value != current:
            result.append({current: _decode(tokenizer, input_ids[start:index])})
            start = index
            current = value
    result.append({current: _decode(tokenizer, input_ids[start:])})
    return result


def _error_detail(stage: str, exc: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "repr": repr(exc),
        "traceback": traceback.format_exc(),
    }


def _write_report(report: dict[str, Any], report_path: str | Path | None) -> Path | None:
    if report_path is None:
        return None
    path = Path(report_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, default=str)
            output.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _template_capabilities(template: Any, tool_case: dict[str, Any] | None = None) -> dict[str, bool]:
    """Report capabilities from the actual registered LlamaFactory template."""
    from llamafactory.data.template import ReasoningTemplate

    supports_thinking = isinstance(template, ReasoningTemplate)
    function_formatter = getattr(template, "format_function", None)
    tools_formatter = getattr(template, "format_tools", None)
    supports_tools = bool(
        function_formatter
        and tools_formatter
        and callable(getattr(function_formatter, "apply", None))
        and callable(getattr(tools_formatter, "apply", None))
        and getattr(function_formatter, "tool_format", None)
        and getattr(tools_formatter, "tool_format", None)
    )
    if supports_tools and tool_case is not None:
        try:
            tool_json = _tools_json(tool_case)
            function_json = json.dumps(
                {"name": "get_current_temperature", "arguments": {"city": "Paris"}},
                ensure_ascii=False,
            )
            tools_formatter.apply(content=tool_json)
            function_formatter.apply(
                content=function_json,
                thought_words=getattr(template, "thought_words", None),
                tool_call_words=getattr(template, "tool_call_words", None),
            )
        except Exception:
            supports_tools = False
    return {"thinking": supports_thinking, "tools": supports_tools}


def _unsupported_reason(
    case: dict[str, Any],
    capabilities: dict[str, bool],
    thinking: bool,
    official_supports_thinking_switch: bool,
) -> str | None:
    if thinking and not capabilities["thinking"]:
        return "thinking_not_supported_by_llamafactory_template"
    if not thinking and capabilities["thinking"] and not official_supports_thinking_switch:
        return "non_thinking_not_supported_by_transformers_template"
    if case.get("tools") and not capabilities["tools"]:
        return "tools_not_supported_by_llamafactory_template"
    return None


def _training_prompt_ids(input_ids: list[int], labels: list[int], inference_ids: list[int] | None = None) -> list[int]:
    target_indexes = [index for index, label in enumerate(labels) if label != -100]
    if not any(label == -100 for label in labels) and inference_ids is not None:
        if input_ids[: len(inference_ids)] == inference_ids:
            return input_ids[: len(inference_ids)]
        common_prefix = 0
        for training_id, inference_id in zip(input_ids, inference_ids):
            if training_id != inference_id:
                break
            common_prefix += 1
        return input_ids[:common_prefix]
    if not target_indexes:
        return input_ids
    start = target_indexes[-1]
    while start > 0 and labels[start - 1] != -100:
        start -= 1
    return input_ids[:start]


def _chat_template_text(tokenizer: Any) -> str:
    value = getattr(tokenizer, "chat_template", "")
    return "\n".join(str(item) for item in value.values()) if isinstance(value, dict) else str(value)


def _select_chat_template(tokenizer: Any, has_tools: bool) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, list):
        chat_template = {
            item["name"]: item["template"]
            for item in chat_template
            if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("template"), str)
        }
    if isinstance(chat_template, dict):
        if has_tools and isinstance(chat_template.get("tool_use"), str):
            return chat_template["tool_use"]
        if isinstance(chat_template.get("default"), str):
            return chat_template["default"]
        if len(chat_template) == 1:
            candidate = next(iter(chat_template.values()))
            if isinstance(candidate, str):
                return candidate
    raise ValueError("The tokenizer does not provide an applicable chat template.")


def _official_template_variables(tokenizer: Any, has_tools: bool = False) -> set[str]:
    from jinja2 import Environment, meta

    try:
        source = _select_chat_template(tokenizer, has_tools=has_tools)
        return meta.find_undeclared_variables(Environment().parse(source))
    except Exception:
        return set()


def _official_supports_tools(tokenizer: Any) -> bool:
    return "tools" in _official_template_variables(tokenizer, has_tools=True)


def _official_uses_message_field(tokenizer: Any, field: str) -> bool:
    from jinja2 import Environment, nodes

    try:
        parsed = Environment().parse(_select_chat_template(tokenizer, has_tools=True))
    except Exception:
        return False

    if any(node.attr == field for node in parsed.find_all(nodes.Getattr)):
        return True
    if any(isinstance(node.arg, nodes.Const) and node.arg.value == field for node in parsed.find_all(nodes.Getitem)):
        return True
    for node in parsed.find_all(nodes.Call):
        if (
            isinstance(node.node, nodes.Getattr)
            and node.node.attr == "get"
            and node.args
            and isinstance(node.args[0], nodes.Const)
            and node.args[0].value == field
        ):
            return True

    return False


def _official_tool_mode(tokenizer: Any) -> str:
    text = _chat_template_text(tokenizer)
    if "assistant_tool_call" in text and "tool_response" in text:
        return "granite3_roles"
    if "metadata" in text and "role == 'observation'" in text:
        return "glm_metadata"
    if "_args.items()" in text or "arguments | tojson" in text or "<arg_key>" in text:
        return "structured_dict"
    if not _official_uses_message_field(tokenizer, "tool_calls"):
        return "serialized_observation" if "role == 'observation'" in text else "serialized_content"
    if "role == 'observation'" in text:
        return "observation"
    return "tool"


def _official_messages(
    case: dict[str, Any], tokenizer: Any, template: Any, include_final: bool, default_system: str | None
) -> list[dict[str, Any]]:
    mode = _official_tool_mode(tokenizer)
    conversations = case["conversations"] if include_final else case["conversations"][:-1]
    messages: list[dict[str, Any]] = []
    pending_name: str | None = None
    for item in conversations:
        role, content = item["from"], item["value"]
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
        elif role == "function_call":
            thought, raw_call = "", content
            if END_THINK in raw_call:
                thought, raw_call = raw_call.rsplit(END_THINK, maxsplit=1)
                thought += END_THINK
            call = json.loads(raw_call.strip())
            pending_name = call["name"]
            arguments = call["arguments"]
            if mode == "glm_metadata":
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                messages.append({"role": "assistant", "content": thought + arguments, "metadata": pending_name})
            elif mode == "granite3_roles":
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                messages.append(
                    {
                        "role": "assistant_tool_call",
                        "content": json.dumps({"name": pending_name, "arguments": arguments}, ensure_ascii=False),
                    }
                )
            elif mode in {"serialized_content", "serialized_observation"}:
                from llamafactory.data.tool_utils import FunctionCall

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_utils = getattr(template.format_function, "tool_utils", None)
                if tool_utils is None:
                    raise ValueError("The LlamaFactory function formatter does not expose tool serialization.")
                tool_call = tool_utils.function_formatter(
                    [FunctionCall(pending_name, json.dumps(arguments, ensure_ascii=False))]
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": thought + tool_call,
                    }
                )
            else:
                if mode != "structured_dict" and not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif mode == "structured_dict" and isinstance(arguments, str):
                    arguments = json.loads(arguments)
                messages.append(
                    {
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": pending_name, "arguments": arguments},
                            }
                        ],
                    }
                )
        elif role == "observation":
            if mode == "granite3_roles":
                messages.append({"role": "tool_response", "content": content})
            else:
                role_name = (
                    "observation" if mode in {"glm_metadata", "observation", "serialized_observation"} else "tool"
                )
                message = {"role": role_name, "content": content}
                if role_name == "tool":
                    message["name"] = pending_name
                messages.append(message)
        else:
            raise ValueError(f"Unsupported ShareGPT role: {role}")
    system = case.get("system") or default_system
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def _lf_messages(case: dict[str, Any], include_final: bool) -> list[dict[str, str]]:
    role_map = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
    conversations = case["conversations"] if include_final else case["conversations"][:-1]
    return [{"role": role_map[item["from"]], "content": item["value"]} for item in conversations]


def _tools_json(case: dict[str, Any]) -> str | None:
    return json.dumps(case["tools"], ensure_ascii=False) if case.get("tools") else None


def _write_embedded_dataset(directory: Path, cases: list[dict[str, Any]]) -> None:
    records = []
    for case in cases:
        records.append(
            {
                "conversations": case["conversations"],
                "system": case.get("system", ""),
                "tools": _tools_json(case) or "",
            }
        )
    (directory / "embedded.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    dataset_info = {
        "embedded": {
            "file_name": "embedded.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        }
    }
    (directory / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")


def _train_runtime(
    model_args: Any,
    template_name: str | None,
    cases: list[dict[str, Any]],
    thinking: bool,
    official_supports_thinking_switch: bool,
    user_data_args: Any | None = None,
    user_training_args: Any | None = None,
) -> tuple[
    list[dict[str, list[int]]],
    Any,
    Any,
    Any,
    Any,
    dict[str, bool],
    list[dict[str, Any]],
    dict[str, str],
]:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.model import load_tokenizer

    with tempfile.TemporaryDirectory(prefix="llamafactory-template-audit-") as temporary:
        dataset_dir = Path(temporary)
        if user_data_args is None:
            data_args = DataArguments(
                template=template_name,
                enable_thinking=thinking,
                cutoff_len=CUTOFF_LEN,
                dataset="embedded",
                dataset_dir=str(dataset_dir),
                preprocessing_num_workers=1,
                preprocessing_batch_size=1,
                overwrite_cache=True,
            )
        else:
            data_args = copy.deepcopy(user_data_args)
            data_args.template = template_name
            data_args.enable_thinking = thinking
            data_args.dataset = ["embedded"]
            data_args.eval_dataset = None
            data_args.dataset_dir = str(dataset_dir)
            data_args.media_dir = str(dataset_dir)
            data_args.streaming = False
            data_args.overwrite_cache = True
            data_args.preprocessing_num_workers = 1
            data_args.preprocessing_batch_size = 1
            data_args.max_samples = None
            data_args.val_size = 0.0
            data_args.tokenized_path = None
            # Packing is a batching concern and would merge multiple audit cases.
            data_args.packing = False
            data_args.neat_packing = False
        training_args = SimpleNamespace(
            dataloader_num_workers=0,
            local_process_index=0,
            main_process_first=lambda **_: nullcontext(),
            output_dir=str(dataset_dir / "output"),
            predict_with_generate=False,
            seed=getattr(user_training_args, "seed", 42),
            should_log=False,
            should_save=False,
        )
        tokenizer_module = load_tokenizer(copy.deepcopy(model_args))
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        tool_case = next((case for case in cases if case.get("tools")), None)
        capabilities = _template_capabilities(template, tool_case)
        unsupported: dict[str, str] = {}
        for case in cases:
            reason = _unsupported_reason(
                case,
                capabilities,
                thinking,
                official_supports_thinking_switch,
            )
            if reason is not None:
                unsupported[case["id"]] = reason
        eligible_cases = [case for case in cases if case["id"] not in unsupported]
        if not eligible_cases:
            return [], model_args, data_args, tokenizer_module, template, capabilities, eligible_cases, unsupported
        _write_embedded_dataset(dataset_dir, eligible_cases)
        dataset_module = get_dataset(
            template,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )
        dataset = dataset_module["train_dataset"]
        training_samples = [
            {
                "input_ids": [int(value) for value in input_row],
                "labels": [int(value) for value in label_row],
            }
            for input_row, label_row in zip(dataset["input_ids"], dataset["labels"], strict=True)
        ]
        return (
            training_samples,
            model_args,
            data_args,
            tokenizer_module,
            template,
            capabilities,
            eligible_cases,
            unsupported,
        )


def _inference_ids(
    tokenizer_module: dict[str, Any],
    template: Any,
    generating_args: Any,
    case: dict[str, Any],
) -> list[int]:
    import torch

    from llamafactory.chat.hf_engine import HuggingfaceEngine

    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.padding_side = "left"
    messages = _lf_messages(case, include_final=False)
    model_stub = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(model_type="prompt_only"),
    )
    generation_kwargs = generating_args.to_dict(obey_generation_config=True)
    # `_process_args` consumes these generation keys even though no generation
    # is started by this prompt-only audit.
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("temperature", 0.0)
    generation_kwargs.setdefault("top_p", 1.0)
    generation_kwargs.setdefault("top_k", 50)
    generation_kwargs.setdefault("repetition_penalty", 1.0)
    generation_kwargs.setdefault("length_penalty", 1.0)
    generation_kwargs.setdefault("skip_special_tokens", False)
    kwargs, _ = HuggingfaceEngine._process_args(
        model_stub,
        tokenizer,
        tokenizer_module.get("processor"),
        template,
        generation_kwargs,
        messages,
        case.get("system") or None,
        _tools_json(case),
        input_kwargs={},
    )
    return _token_ids(kwargs["inputs"])


def _official_ids(
    tokenizer: Any,
    template: Any,
    case: dict[str, Any],
    thinking: bool,
    default_system: str | None,
    include_final: bool,
    add_generation_prompt: bool,
) -> list[int]:
    messages = _official_messages(case, tokenizer, template, include_final, default_system)
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    parameters = inspect.signature(tokenizer.apply_chat_template).parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if "chat_template_kwargs" in parameters:
        kwargs["chat_template_kwargs"] = {"enable_thinking": thinking}
    elif "enable_thinking" in parameters or accepts_kwargs:
        kwargs["enable_thinking"] = thinking
    if case.get("tools") and ("tools" in parameters or accepts_kwargs):
        kwargs["tools"] = case["tools"]
    return _token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _case_report(
    case: dict[str, Any],
    training_sample: dict[str, list[int]],
    lf_infer: list[int],
    tokenizer: Any,
    template: Any,
    official_tokenizer: Any,
    default_system: str | None,
) -> dict[str, Any]:
    thinking = bool(case["enable_thinking"])
    official_infer = _official_ids(official_tokenizer, template, case, thinking, default_system, False, True)
    input_ids = training_sample["input_ids"]
    labels = training_sample["labels"]
    lf_prompt_ids = _training_prompt_ids(input_ids, labels, lf_infer)
    loss_participation = _loss_participation(tokenizer, input_ids, labels)
    comparisons = {
        "training_prompt_vs_llamafactory_inference": _compare_ids(lf_prompt_ids, lf_infer),
        "training_prompt_vs_transformers_inference": _compare_ids(lf_prompt_ids, official_infer),
        "llamafactory_inference_vs_transformers_inference": _compare_ids(lf_infer, official_infer),
    }
    return {
        "id": case["id"],
        "status": "same" if all(item == "same" for item in comparisons.values()) else "different",
        "actual_input": copy.deepcopy(case),
        "llamafactory_training_complete_actual_content": _decode(tokenizer, input_ids),
        "loss_participation": loss_participation,
        "llamafactory_training_prompt_actual_content": _decode(tokenizer, lf_prompt_ids),
        "llamafactory_inference_actual_input": _decode(tokenizer, lf_infer),
        "transformers_inference_actual_input": _decode(official_tokenizer, official_infer),
        "comparisons": comparisons,
    }


def _overall_status(results: list[dict[str, Any]], setup_error: dict[str, Any] | None = None) -> str:
    if setup_error is not None or any(result.get("status") == "error" for result in results):
        return "error"
    if any(result.get("status") == "different" for result in results):
        return "different"
    return "same" if results else "not_applicable"


def _run_audit(
    model_args: Any,
    data_args: Any,
    training_args: Any,
    generating_args: Any,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer

    template_name = getattr(data_args, "template", None)
    official_tokenizer = load_tokenizer(copy.deepcopy(model_args))["tokenizer"]
    official_supports_tools = _official_supports_tools(official_tokenizer)
    official_supports_thinking_switch = "enable_thinking" in _official_template_variables(official_tokenizer)
    official_template_text = _chat_template_text(official_tokenizer)
    official_supports_thinking = official_supports_thinking_switch or any(
        marker in official_template_text for marker in ("<think>", "</think>", "reasoning_content")
    )
    official_unsupported = {case["id"] for case in cases if case.get("tools") and not official_supports_tools}
    groups: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for case in cases:
        if case["id"] not in official_unsupported:
            groups.setdefault((case["enable_thinking"], bool(case.get("tools"))), []).append(case)

    lf_training: dict[str, dict[str, list[int]]] = {}
    lf_inference: dict[str, list[int]] = {}
    lf_tokenizers: dict[str, Any] = {}
    lf_templates: dict[str, Any] = {}
    lf_unsupported: set[str] = set()
    case_errors: dict[str, dict[str, str]] = {}
    supports_thinking = False
    supports_tools = False
    for (thinking, _), group_cases in groups.items():
        try:
            with _suppress_runtime_output():
                (
                    training_samples,
                    _,
                    runtime_data_args,
                    _,
                    _,
                    capabilities,
                    eligible_cases,
                    unsupported,
                ) = _train_runtime(
                    model_args,
                    template_name,
                    group_cases,
                    thinking,
                    official_supports_thinking_switch,
                    data_args,
                    training_args,
                )
            supports_thinking = supports_thinking or capabilities["thinking"]
            supports_tools = supports_tools or capabilities["tools"]
            lf_unsupported.update(unsupported)
            if not eligible_cases:
                continue
            with _suppress_runtime_output():
                infer_tokenizer_module = load_tokenizer(copy.deepcopy(model_args))
                infer_tokenizer = infer_tokenizer_module["tokenizer"]
                infer_tokenizer.padding_side = "left"
                infer_template = get_template_and_fix_tokenizer(infer_tokenizer, runtime_data_args)
            for case, training_sample in zip(eligible_cases, training_samples, strict=True):
                case_id = case["id"]
                lf_training[case_id] = training_sample
                lf_tokenizers[case_id] = infer_tokenizer
                lf_templates[case_id] = infer_template
                try:
                    with _suppress_runtime_output():
                        lf_inference[case_id] = _inference_ids(
                            infer_tokenizer_module,
                            infer_template,
                            generating_args,
                            case,
                        )
                except Exception as exc:
                    case_errors[case_id] = _error_detail("llamafactory_inference", exc)
        except Exception as exc:
            error = _error_detail("llamafactory_training", exc)
            for case in group_cases:
                case_errors[case["id"]] = error

    results: list[dict[str, Any]] = []
    default_system = getattr(data_args, "default_system", None)
    for case in cases:
        case_id = case["id"]
        if case_id in official_unsupported or case_id in lf_unsupported:
            continue
        if case_id in case_errors or case_id not in lf_training or case_id not in lf_inference:
            results.append(
                {
                    "id": case_id,
                    "status": "error",
                    "actual_input": copy.deepcopy(case),
                    "error": case_errors.get(
                        case_id,
                        {
                            "stage": "llamafactory_runtime",
                            "type": "RuntimeError",
                            "message": "LlamaFactory runtime did not return the case.",
                            "repr": "RuntimeError('LlamaFactory runtime did not return the case.')",
                            "traceback": "",
                        },
                    ),
                }
            )
            continue
        try:
            results.append(
                _case_report(
                    case,
                    lf_training[case_id],
                    lf_inference[case_id],
                    lf_tokenizers[case_id],
                    lf_templates[case_id],
                    official_tokenizer,
                    default_system,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "id": case_id,
                    "status": "error",
                    "actual_input": copy.deepcopy(case),
                    "error": _error_detail("build_case_report", exc),
                }
            )

    return {
        "status": _overall_status(results),
        "supports_thinking": supports_thinking and official_supports_thinking,
        "supports_tool_calls": supports_tools and official_supports_tools,
        "cases": results,
    }


def audit_template_consistency(
    model_args: Any,
    data_args: Any,
    training_args: Any,
    generating_args: Any | None = None,
    *,
    cases: Iterable[dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one LlamaFactory training configuration using its real argument objects."""
    if generating_args is None:
        from llamafactory.hparams import GeneratingArguments

        generating_args = GeneratingArguments(
            do_sample=False,
            temperature=0.0,
            skip_special_tokens=False,
            max_new_tokens=1,
        )
    if cases is None:
        from llamafactory.data.template import TEMPLATES, ReasoningTemplate

        template = TEMPLATES.get(getattr(data_args, "template", None))
        requested_thinking = bool(getattr(data_args, "enable_thinking", False))
        modes = (
            {False, requested_thinking}
            if template is None
            else {requested_thinking and isinstance(template, ReasoningTemplate)}
        )
        selected_cases = [copy.deepcopy(case) for case in CASES if bool(case["enable_thinking"]) in modes]
    else:
        selected_cases = [copy.deepcopy(case) for case in cases]
    report: dict[str, Any] = {
        "model_name_or_path": getattr(model_args, "model_name_or_path", None),
        "template": getattr(data_args, "template", None),
        "cases": [],
    }
    try:
        with _suppress_runtime_output():
            report.update(_run_audit(model_args, data_args, training_args, generating_args, selected_cases))
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "supports_thinking": False,
                "supports_tool_calls": False,
                "setup_error": _error_detail("prepare_llamafactory_runtime", exc),
            }
        )
    if report_path is not None:
        report["report_path"] = str(Path(report_path).expanduser().resolve())
    _write_report(report, report_path)
    return report


def _consistency_level(report: dict[str, Any]) -> str | None:
    cases = report.get("cases", [])
    if any(
        case.get("comparisons", {}).get("llamafactory_inference_vs_transformers_inference") == "different"
        for case in cases
    ):
        return "SEVERE"
    if report.get("status") == "error" or any(case.get("status") == "error" for case in cases):
        return None
    if any(any(value != "same" for value in case.get("comparisons", {}).values()) for case in cases):
        return "CAUTION"
    return "NORMAL" if cases else None


def _sample_dataset_capabilities(model_args: Any, data_args: Any, training_args: Any) -> dict[str, Any]:
    result = {"thinking": False, "tools": False, "sampled": 0}
    if not getattr(data_args, "dataset", None):
        return result

    can_interrupt = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    previous_handler = None
    deadline = time.monotonic() + SAMPLE_TIMEOUT

    def handle_timeout(*_: Any) -> None:
        raise TimeoutError

    try:
        if can_interrupt:
            previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, SAMPLE_TIMEOUT)

        from llamafactory.data.loader import _load_single_dataset
        from llamafactory.data.parser import get_dataset_list

        sample_data_args = copy.deepcopy(data_args)
        sample_data_args.streaming = True
        sample_data_args.max_samples = None
        sample_data_args.preprocessing_num_workers = None
        sample_training_args = SimpleNamespace(dataloader_num_workers=1, local_process_index=0)
        seed = getattr(training_args, "seed", 42)
        dataset_attrs = get_dataset_list(sample_data_args.dataset, sample_data_args.dataset_dir)
        for index, dataset_attr in enumerate(dataset_attrs):
            dataset = _load_single_dataset(dataset_attr, model_args, sample_data_args, sample_training_args)
            iterator = iter(dataset.shuffle(buffer_size=1000, seed=seed + index))
            datasets_left = len(dataset_attrs) - index
            samples_left = SAMPLE_SIZE - result["sampled"]
            sample_count = (samples_left + datasets_left - 1) // datasets_left
            for _ in range(sample_count):
                if time.monotonic() >= deadline:
                    raise TimeoutError
                try:
                    example = next(iterator)
                except StopIteration:
                    break

                text = json.dumps(example, ensure_ascii=False, default=str)
                tools = example.get("_tools")
                has_tools = bool(tools) and (
                    not isinstance(tools, str) or tools.strip() not in {"", "[]", "{}", "null"}
                )
                result["sampled"] += 1
                result["thinking"] = result["thinking"] or THINK in text
                result["tools"] = (
                    result["tools"]
                    or any(
                        marker in text
                        for marker in (
                            '"role": "function"',
                            '"role": "function_call"',
                            '"role": "observation"',
                            '"role": "tool"',
                            '"tool_calls"',
                        )
                    )
                    or has_tools
                )
    except Exception:
        pass
    finally:
        if can_interrupt:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    return result


def _capability_warnings(report: dict[str, Any], dataset_capabilities: dict[str, Any]) -> list[str]:
    warnings = []
    if dataset_capabilities["thinking"] and not report.get("supports_thinking", False):
        warnings.append("The sampled training data uses thinking, but the model's templates do not support it.")
    if dataset_capabilities["tools"] and not report.get("supports_tool_calls", False):
        warnings.append("The sampled training data uses tool calls, but the model's templates do not support them.")
    return warnings


def _level_message(level: str | None, capability_warnings: list[str], report_path: Path) -> str:
    if level == "NORMAL":
        detail = "LLaMAFactory inference, Transformers inference, and the training prompt are fully consistent."
    elif level == "CAUTION":
        detail = (
            "LLaMAFactory inference and Transformers inference are consistent, but the training prompt differs "
            "from LLaMAFactory inference. This may be caused by multi-turn context cleanup between training and "
            "inference and may not affect training."
        )
    elif level == "SEVERE":
        detail = (
            "LLaMAFactory inference differs from Transformers inference. A severe train-inference inconsistency "
            "was detected."
        )
    else:
        return f"Template consistency check could not be completed. Training will continue. Full report: {report_path}"
    if capability_warnings:
        detail += " Capability warning: " + " ".join(capability_warnings)
    return f"Template consistency check: {level}. {detail} Full report: {report_path}"


def run_template_consistency_check(
    model_args: Any,
    data_args: Any,
    training_args: Any,
    generating_args: Any,
) -> dict[str, Any] | None:
    """Run and log a non-blocking template consistency check before training."""
    if (
        not getattr(data_args, "check_template_consistency", True)
        or not getattr(training_args, "do_train", False)
        or not getattr(training_args, "should_save", True)
    ):
        return None

    report_path = Path(training_args.output_dir).expanduser().resolve() / REPORT_NAME
    try:
        with _suppress_runtime_output():
            dataset_capabilities = _sample_dataset_capabilities(model_args, data_args, training_args)
        report = audit_template_consistency(model_args, data_args, training_args, generating_args)
        level = _consistency_level(report)
        capability_warnings = _capability_warnings(report, dataset_capabilities)
        report.update(
            {
                "level": level,
                "dataset_capabilities": dataset_capabilities,
                "capability_warnings": capability_warnings,
                "report_path": str(report_path),
            }
        )
        _write_report(report, report_path)
        message = _level_message(level, capability_warnings, report_path)
        if level == "NORMAL" and not capability_warnings:
            logger.info_rank0(message)
        else:
            logger.warning_rank0(message)
        return report
    except Exception as exc:
        report = {
            "status": "error",
            "level": None,
            "model_name_or_path": getattr(model_args, "model_name_or_path", None),
            "template": getattr(data_args, "template", None),
            "cases": [],
            "report_path": str(report_path),
            "setup_error": _error_detail("run_training_check", exc),
        }
        try:
            _write_report(report, report_path)
            path_message = f" Full report: {report_path}"
        except Exception:
            path_message = f" The report could not be written to {report_path}."
        logger.warning_rank0(
            f"Template consistency check failed and training will continue.{path_message} Error: {exc}"
        )
        return report

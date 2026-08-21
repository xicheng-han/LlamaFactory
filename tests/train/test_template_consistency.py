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

import json
from types import SimpleNamespace

import pytest

from llamafactory.train import template_consistency, tuner


def _case_report(
    training_vs_llamafactory: str = "same",
    training_vs_transformers: str = "same",
    llamafactory_vs_transformers: str = "same",
) -> dict:
    comparisons = {
        "training_prompt_vs_llamafactory_inference": training_vs_llamafactory,
        "training_prompt_vs_transformers_inference": training_vs_transformers,
        "llamafactory_inference_vs_transformers_inference": llamafactory_vs_transformers,
    }
    return {
        "id": "case",
        "status": "same" if all(value == "same" for value in comparisons.values()) else "different",
        "comparisons": comparisons,
    }


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (_case_report(), "NORMAL"),
        (_case_report("different", "different", "same"), "CAUTION"),
        (_case_report("different", "different", "different"), "SEVERE"),
    ],
)
def test_consistency_levels(case: dict, expected: str):
    assert template_consistency._consistency_level({"status": case["status"], "cases": [case]}) == expected


def test_training_check_writes_report_and_capability_warnings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    messages = []
    report = {
        "status": "same",
        "supports_thinking": False,
        "supports_tool_calls": False,
        "cases": [_case_report()],
    }
    monkeypatch.setattr(template_consistency, "audit_template_consistency", lambda *args: report)
    monkeypatch.setattr(
        template_consistency,
        "_sample_dataset_capabilities",
        lambda *args: {"thinking": True, "tools": True, "sampled": 100},
    )
    monkeypatch.setattr(template_consistency.logger, "info_rank0", messages.append)
    monkeypatch.setattr(template_consistency.logger, "warning_rank0", messages.append)

    result = template_consistency.run_template_consistency_check(
        SimpleNamespace(model_name_or_path="model"),
        SimpleNamespace(
            dataset=None,
            dataset_dir=str(tmp_path),
            check_template_consistency=True,
            template="template",
        ),
        SimpleNamespace(do_train=True, output_dir=str(tmp_path), should_save=True),
        SimpleNamespace(),
    )

    report_path = tmp_path / template_consistency.REPORT_NAME
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == saved_report
    assert saved_report["level"] == "NORMAL"
    assert saved_report["dataset_capabilities"] == {"thinking": True, "tools": True, "sampled": 100}
    assert len(saved_report["capability_warnings"]) == 2
    assert len(messages) == 1
    assert messages[0].count(str(report_path)) == 1
    assert "Template consistency check: NORMAL" in messages[0]
    assert "sampled training data uses thinking" in messages[0]
    assert "sampled training data uses tool calls" in messages[0]


def test_dataset_capability_sample_is_limited_to_100(monkeypatch: pytest.MonkeyPatch):
    from llamafactory.data import loader, parser

    class SampleDataset:
        def shuffle(self, **kwargs):
            return self

        def __iter__(self):
            for _ in range(150):
                yield {
                    "_prompt": [{"role": "user", "content": "<think>reasoning</think>"}],
                    "_response": [{"role": "function", "content": "call"}],
                    "_tools": "",
                }

    monkeypatch.setattr(parser, "get_dataset_list", lambda *args: [SimpleNamespace()])
    monkeypatch.setattr(loader, "_load_single_dataset", lambda *args: SampleDataset())

    result = template_consistency._sample_dataset_capabilities(
        SimpleNamespace(),
        SimpleNamespace(dataset=["dataset"], dataset_dir="data"),
        SimpleNamespace(seed=42),
    )

    assert result == {"thinking": True, "tools": True, "sampled": 100}


def test_training_check_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        template_consistency,
        "audit_template_consistency",
        lambda *args: pytest.fail("audit should not run"),
    )

    result = template_consistency.run_template_consistency_check(
        SimpleNamespace(),
        SimpleNamespace(check_template_consistency=False),
        SimpleNamespace(do_train=True, should_save=True),
        SimpleNamespace(),
    )

    assert result is None


def test_training_check_failure_does_not_raise(tmp_path, monkeypatch: pytest.MonkeyPatch):
    messages = []

    def raise_error(*args):
        raise RuntimeError("audit failed")

    monkeypatch.setattr(template_consistency, "audit_template_consistency", raise_error)
    monkeypatch.setattr(template_consistency.logger, "warning_rank0", messages.append)

    result = template_consistency.run_template_consistency_check(
        SimpleNamespace(model_name_or_path="model"),
        SimpleNamespace(
            check_template_consistency=True,
            dataset=None,
            dataset_dir=str(tmp_path),
            template="template",
        ),
        SimpleNamespace(do_train=True, output_dir=str(tmp_path), should_save=True),
        SimpleNamespace(),
    )

    assert result["status"] == "error"
    assert (tmp_path / template_consistency.REPORT_NAME).is_file()
    assert "training will continue" in messages[0]


def test_training_entry_runs_check_before_training(monkeypatch: pytest.MonkeyPatch):
    events = []
    model_args = SimpleNamespace()
    data_args = SimpleNamespace()
    training_args = SimpleNamespace(do_train=True)
    finetuning_args = SimpleNamespace(
        early_stopping_steps=None,
        pissa_convert=False,
        stage="sft",
        use_hyper_parallel=False,
        use_mca=False,
        use_megatron_bridge=False,
        use_swanlab=False,
    )
    generating_args = SimpleNamespace()
    monkeypatch.setattr(
        tuner,
        "get_train_args",
        lambda args: (model_args, data_args, training_args, finetuning_args, generating_args),
    )
    monkeypatch.setattr(tuner, "LogCallback", lambda: object())
    monkeypatch.setattr(tuner, "ReporterCallback", lambda *args: object())
    monkeypatch.setattr(tuner, "run_template_consistency_check", lambda *args: events.append("check"))
    monkeypatch.setattr(tuner, "run_sft", lambda *args: events.append("train"))
    monkeypatch.setattr(tuner, "is_ray_available", lambda: False)

    tuner._training_function({"args": {}, "callbacks": []})

    assert events == ["check", "train"]

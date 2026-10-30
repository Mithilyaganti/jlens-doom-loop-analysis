"""Licenses checked for the prompt sources.

Dates are the day the upstream text was read. They are records of what
this package relied on, not a claim that the license cannot change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LicenseRecord:
    dataset_id: str
    title: str
    license_name: str
    license_url: str
    source_url: str
    citation: str
    redistributable: bool
    notes: str
    checked_on: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "source_url": self.source_url,
            "citation": self.citation,
            "redistributable": self.redistributable,
            "notes": self.notes,
            "checked_on": self.checked_on,
        }


LICENSES: dict[str, LicenseRecord] = {
    "mmlu-prox-lite": LicenseRecord(
        dataset_id="mmlu-prox-lite",
        title="MMLU-ProX-Lite",
        license_name="MIT",
        license_url="https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite",
        source_url="https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite",
        citation="Xuan et al., 2025, MMLU-ProX, arXiv:2503.10497.",
        redistributable=True,
        notes=(
            "The Hugging Face dataset card states that MMLU-ProX is released "
            "under the MIT License. The dataset repo listing on the day this "
            "was checked contained README.md and parquet shards, and no "
            "separate LICENSE file. Upstream MMLU-Pro (TIGER-Lab/MMLU-Pro, "
            "Wang et al. 2024, arXiv:2406.01574) is Apache-2.0. The selected "
            "prompts are the MMLU-ProX-Lite translations. Both notices are "
            "kept because the translation release and the underlying benchmark "
            "do not name the same license."
        ),
        checked_on="2026-10-30",
    ),
    "mmlu-pro": LicenseRecord(
        dataset_id="mmlu-pro",
        title="MMLU-Pro",
        license_name="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        citation="Wang et al., 2024, MMLU-Pro, arXiv:2406.01574.",
        redistributable=True,
        notes="Upstream of MMLU-ProX. Recorded so the prompt notices name it. Not loaded directly.",
        checked_on="2026-10-30",
    ),
    "mgsm": LicenseRecord(
        dataset_id="mgsm",
        title="Multilingual Grade School Math (MGSM)",
        license_name="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://github.com/google-research/url-nlp/tree/main/mgsm",
        citation="Shi et al., 2022, Language models are multilingual chain-of-thought reasoners, arXiv:2210.03057.",
        redistributable=True,
        notes=(
            "The LICENSE file in google-research/url-nlp/mgsm is the Creative "
            "Commons Attribution 4.0 International license (CC-BY-4.0), read "
            "from the raw file on 2026-10-30. A third-party catalog page that "
            "says CC-BY-SA-4.0 was not used. English items come from GSM8K "
            "(Cobbe et al. 2021, arXiv:2110.14168), whose own dataset license "
            "is MIT; the copy in the MGSM repository is covered by that "
            "directory's CC-BY-4.0 file. MGSM languages are English, Spanish, "
            "French, German, Russian, Chinese, Japanese, Thai, Swahili, "
            "Bengali, and Telugu. Hindi is not in the benchmark, so MGSM "
            "items are not part of the trilingual matched 300."
        ),
        checked_on="2026-10-30",
    ),
    "gsm8k": LicenseRecord(
        dataset_id="gsm8k",
        title="GSM8K",
        license_name="MIT",
        license_url="https://github.com/openai/grade-school-math/blob/master/LICENSE",
        source_url="https://github.com/openai/grade-school-math",
        citation="Cobbe et al., 2021, Training Verifiers to Solve Math Word Problems, arXiv:2110.14168.",
        redistributable=True,
        notes="Source of the 250 English problems MGSM translates. Not loaded as its own split.",
        checked_on="2026-10-30",
    ),
}


def require_known(dataset_id: str) -> LicenseRecord:
    try:
        return LICENSES[dataset_id]
    except KeyError as exc:
        raise KeyError(f"no license record for {dataset_id!r}") from exc

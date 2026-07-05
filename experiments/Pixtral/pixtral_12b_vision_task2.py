import argparse
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys

import torch
from transformers.utils import logging as hf_logging

from mistralai import Mistral
from evaluation.llm_metrics import (
    calculate_bertscore,
    calculate_information_preservation,
    calculate_rouge_scores,
)
from llm_processing.prompts import PROMPTS, get_prompt

# Suppress known harmless HF load warnings.
hf_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    if (cwd.parent / "src").exists():
        return cwd.parent
    return cwd.parent.parent


def resolve_mistral_api_key() -> str:

    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        return env_key
    raise EnvironmentError("MISTRAL_API_KEY is not set. Export your Mistral API key.")


class PixtralVisionTask2Pipeline:
    def __init__(
        self,
        client,
        system_prompt,
        model_name="pixtral-12b-2409",
        max_tokens=1024,
        top_p=1.0,
    ):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt

    def _encode_image(self, image_path):
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        return self._encode_image_bytes(image_bytes, mime_type)

    def _encode_image_bytes(self, image_bytes, mime_type):
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _normalize_content(self, content):
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(p.strip() for p in parts if p).strip()
        return str(content).strip() if content is not None else ""

    def process_image(self, image_path, user_prompt):
        image_url = self._encode_image(image_path)
        response = self.client.chat.complete(
            model=self.model_name,
            temperature=0.0,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": image_url},
                    ],
                },
            ],
            random_seed=42,
        )
        return self._normalize_content(response.choices[0].message.content)

    def process_image_bytes(self, image_bytes, user_prompt, mime_type="image/png"):
        image_url = self._encode_image_bytes(image_bytes, mime_type)
        response = self.client.chat.complete(
            model=self.model_name,
            temperature=0.0,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": image_url},
                    ],
                },
            ],
            random_seed=42,
        )
        return self._normalize_content(response.choices[0].message.content)


def load_scenarios(scenarios_file: Path):
    with open(scenarios_file, encoding="utf-8") as handle:
        scenarios = json.load(handle)
    return {s["patient"]: s for s in scenarios if "patient" in s}


def build_single_metadata(image_path: Path, label_path: Path):
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(label_path, encoding="utf-8") as handle:
        lbl = json.load(handle)
    gen_params = lbl.get("generation_parameters", {})
    font_file = os.path.basename(gen_params.get("font_path", ""))
    return {
        "sample_id": image_path.stem,
        "image_path": image_path,
        "label_path": str(label_path),
        "font": os.path.splitext(font_file)[0] if font_file else "unknown",
        "scenario": lbl.get("patient", ""),
        "full_text": lbl.get("full_text", ""),
    }


def build_metadata_from_label(label_path: Path, sample_id=""):
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with open(label_path, encoding="utf-8") as handle:
        lbl = json.load(handle)
    gen_params = lbl.get("generation_parameters", {})
    font_file = os.path.basename(gen_params.get("font_path", ""))
    return {
        "sample_id": sample_id or "image",
        "label_path": str(label_path),
        "font": os.path.splitext(font_file)[0] if font_file else "unknown",
        "scenario": lbl.get("patient", ""),
        "full_text": lbl.get("full_text", ""),
    }


def run_single(meta, scenario_lookup, pipeline, user_prompt):
    curr_scenario = scenario_lookup.get(meta["scenario"], {})
    ref_summary = curr_scenario.get("reference_summary", "")
    if not ref_summary:
        print("SKIPPED - missing reference_summary")
        return None

    try:
        summary_out = pipeline.process_image(
            str(meta["image_path"]), user_prompt=user_prompt
        )
    except Exception as exc:
        print(f"VLM ERROR: {exc}")
        return None

    if not summary_out or not summary_out.strip():
        print("SKIPPED - empty model output")
        return None

    rouge = calculate_rouge_scores(reference=ref_summary, hypothesis=summary_out)
    bert = calculate_bertscore(reference=ref_summary, hypothesis=summary_out)
    pres = calculate_information_preservation(
        reference=ref_summary, summary=summary_out
    )

    return {
        "sample_id": meta["sample_id"],
        "font": meta["font"],
        "scenario": meta["scenario"],
        "output": summary_out,
        **rouge,
        **bert,
        **pres,
    }


def run_single_bytes(
    meta, scenario_lookup, pipeline, user_prompt, image_bytes, mime_type="image/png"
):
    curr_scenario = scenario_lookup.get(meta["scenario"], {})
    ref_summary = curr_scenario.get("reference_summary", "")
    if not ref_summary:
        print("SKIPPED - missing reference_summary")
        return None

    try:
        summary_out = pipeline.process_image_bytes(
            image_bytes, user_prompt=user_prompt, mime_type=mime_type
        )
    except Exception as exc:
        print(f"VLM ERROR: {exc}")
        return None

    if not summary_out or not summary_out.strip():
        print("SKIPPED - empty model output")
        return None

    rouge = calculate_rouge_scores(reference=ref_summary, hypothesis=summary_out)
    bert = calculate_bertscore(reference=ref_summary, hypothesis=summary_out)
    pres = calculate_information_preservation(
        reference=ref_summary, summary=summary_out
    )

    return {
        "sample_id": meta["sample_id"],
        "font": meta["font"],
        "scenario": meta["scenario"],
        "output": summary_out,
        **rouge,
        **bert,
        **pres,
    }


def evaluate_single_image(
    image_path,
    label_path,
    prompt_name="VLM_prompt_v2",
    model_name="pixtral-12b-2409",
    max_tokens=1024,
    top_p=1.0,
    api_key=None,
    project_root=None,
):
    project_root = project_root or resolve_project_root()
    sys.path.insert(0, str(project_root / "src"))

    scenarios_file = project_root / "data" / "synthetic" / "scenarios.json"
    scenario_lookup = load_scenarios(scenarios_file)

    prompt_template = get_prompt(prompt_name)
    user_prompt = (
        prompt_template["task"]
        .replace("Text:\n{text}", "")
        .replace("{text}", "")
        .strip()
    )
    system_prompt = prompt_template["system"]

    client = Mistral(api_key=resolve_mistral_api_key(api_key))
    pipeline = PixtralVisionTask2Pipeline(
        client=client,
        system_prompt=system_prompt,
        model_name=model_name,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    meta = build_single_metadata(Path(image_path), Path(label_path))
    return run_single(meta, scenario_lookup, pipeline, user_prompt)


def evaluate_single_image_bytes(
    image_bytes,
    label_path,
    prompt_name="VLM_prompt_v2",
    model_name="pixtral-12b-2409",
    max_tokens=1024,
    top_p=1.0,
    api_key=None,
    project_root=None,
    mime_type="image/png",
    sample_id="",
):
    project_root = project_root or resolve_project_root()
    sys.path.insert(0, str(project_root / "src"))

    scenarios_file = project_root / "data" / "synthetic" / "scenarios.json"
    scenario_lookup = load_scenarios(scenarios_file)

    prompt_template = get_prompt(prompt_name)
    user_prompt = (
        prompt_template["task"]
        .replace("Text:\n{text}", "")
        .replace("{text}", "")
        .strip()
    )
    system_prompt = prompt_template["system"]

    client = Mistral(api_key=resolve_mistral_api_key(api_key))
    pipeline = PixtralVisionTask2Pipeline(
        client=client,
        system_prompt=system_prompt,
        model_name=model_name,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    meta = build_metadata_from_label(Path(label_path), sample_id=sample_id)
    return run_single_bytes(
        meta, scenario_lookup, pipeline, user_prompt, image_bytes, mime_type=mime_type
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pixtral-8B Task 2 end-to-end evaluation"
    )
    parser.add_argument(
        "--prompt", default="VLM_prompt_v2", choices=sorted(PROMPTS.keys())
    )
    parser.add_argument("--model", default="pixtral-12b-2409")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image", default="", help="Evaluate a single image path")
    parser.add_argument("--label", default="", help="Label JSON for single image")
    return parser.parse_args()


def main():
    project_root = resolve_project_root()
    sys.path.insert(0, str(project_root / "src"))

    scenarios_file = project_root / "data" / "synthetic" / "scenarios.json"
    args = parse_args()

    api_key = resolve_mistral_api_key()
    scenario_lookup = load_scenarios(scenarios_file)
    print(f"Loaded {len(scenario_lookup)} scenarios with reference_summary\n")

    prompt_template = get_prompt(args.prompt)
    user_prompt = (
        prompt_template["task"]
        .replace("Text:\n{text}", "")
        .replace("{text}", "")
        .strip()
    )
    system_prompt = prompt_template["system"]

    client = Mistral(api_key=api_key)
    pipeline = PixtralVisionTask2Pipeline(
        client=client,
        system_prompt=system_prompt,
        model_name=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    if not args.image or not args.label:
        raise ValueError("--image and --label are required")

    single_meta = build_single_metadata(Path(args.image), Path(args.label))
    print("Single-image evaluation")
    print(f"  Image: {single_meta['image_path']}")
    print(f"  Label: {single_meta['label_path']}")
    print(f"  Scenario: {single_meta['scenario'] or 'unknown'}")

    result = run_single(single_meta, scenario_lookup, pipeline, user_prompt)
    if result:
        print(
            f"\n{result['sample_id']:<22} "
            f"{result['rouge1_f1']:>6.1f}% "
            f"{result['bertscore_f1']:>8.1f}% "
            f"{result['medical_term_f1']:>7.1f}%"
        )


if __name__ == "__main__":
    print("\nPixtral-8B Task 2 pipeline")
    print(f"Torch: {torch.__version__}")
    print("HF model-loading warnings for known head-mismatch are suppressed")
    main()

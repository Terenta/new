from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import project_root, resolve_path
from .exceptions import ConfigError


@dataclass
class CommandSpec:
    phase: str
    cwd: str
    argv: list[str]
    output: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _roots() -> tuple[Path, Path]:
    root = project_root()
    vendor = Path(os.getenv("CVF_VENDOR_ROOT", str(root / "vendor"))).expanduser().resolve()
    models = Path(os.getenv("CVF_MODELS_ROOT", "/opt/cvf-models")).expanduser().resolve()
    return vendor, models


def _backend_python(name: str) -> str:
    root = project_root()
    env_names = {
        "scail2": "CVF_SCAIL2_PYTHON",
        "scail2_pose": "CVF_SCAIL2_POSE_PYTHON",
        "wan22": "CVF_WAN22_PYTHON",
        "wan_animate2": "CVF_WAN_ANIMATE2_PYTHON",
        "rvc": "CVF_RVC_PYTHON",
    }
    configured = os.getenv(env_names[name])
    if configured:
        return str(Path(configured).expanduser())
    if os.name == "nt":
        return sys.executable
    return str(root / ".envs" / name.replace("_", "-") / "bin" / "python")


def _prepared(job: dict[str, Any]) -> dict[str, Path]:
    inputs = job.get("inputs", {})
    required = ("reference_image", "reference_mask", "driving_video", "driving_mask")
    missing = [name for name in required if not inputs.get(name)]
    if missing:
        raise ConfigError("Job is missing inputs: " + ", ".join(missing))
    return {name: resolve_path(inputs[name]) for name in required}


def scail2_preprocess_command(job: dict[str, Any]) -> CommandSpec:
    vendor, models = _roots()
    prepared_dir = resolve_path(job["inputs"]["driving_video"]).parent
    script = vendor / "SCAIL-2" / "SCAIL-Pose" / "NLFPoseExtract" / "process_animation_aio.py"
    return CommandSpec(
        phase="preprocess_masks",
        cwd=str(vendor / "SCAIL-2" / "SCAIL-Pose"),
        argv=[
            _backend_python("scail2_pose"),
            str(script),
            "--subdir",
            str(prepared_dir),
            "--video_name",
            "driving.mp4",
            "--e2e_mode",
            "--max_persons",
            "1",
            "--sam3_model",
            str(models / "sam3" / "sam3.pt"),
        ],
    )


def scail2_commands(job: dict[str, Any], smoke: bool = False) -> list[CommandSpec]:
    vendor, models = _roots()
    inputs = _prepared(job)
    generation = job.get("generation", {})
    output_root = resolve_path(job.get("output", {}).get("root", "artifacts/pilot/runs/unknown"))
    seeds = list(generation.get("seeds", [1001]))
    if smoke:
        seeds = seeds[:1]
    specs: list[CommandSpec] = []
    for seed in seeds:
        output = output_root / ("smoke" if smoke else "raw") / "scail2" / f"seed_{seed}.mp4"
        specs.append(
            CommandSpec(
                phase="render_smoke" if smoke else "render",
                cwd=str(vendor / "SCAIL-2"),
                output=str(output),
                argv=[
                    _backend_python("scail2"),
                    str(vendor / "SCAIL-2" / "generate.py"),
                    "--model",
                    "SCAIL-14B",
                    "--ckpt_dir",
                    str(models / "SCAIL-2"),
                    "--scail_path",
                    str(models / "SCAIL-2" / "SCAIL-2.safetensors"),
                    "--target_w",
                    str(generation.get("width", 704)),
                    "--target_h",
                    str(generation.get("height", 1280)),
                    "--image",
                    str(inputs["reference_image"]),
                    "--mask_image",
                    str(inputs["reference_mask"]),
                    "--pose",
                    str(inputs["driving_video"]),
                    "--mask_video",
                    str(inputs["driving_mask"]),
                    "--prompt",
                    str(generation.get("prompt", "")),
                    "--base_seed",
                    str(seed),
                    "--sample_steps",
                    str(generation.get("sample_steps", 40)),
                    "--sample_shift",
                    str(generation.get("sample_shift", 3.0)),
                    "--sample_guide_scale",
                    str(generation.get("guidance_scale", 5.0)),
                    "--sample_solver",
                    str(generation.get("solver", "unipc")),
                    "--segment_len",
                    str(generation.get("window_frames", 77)),
                    "--segment_overlap",
                    str(generation.get("overlap_frames", 8)),
                    "--save_file",
                    str(output),
                ],
            )
        )
    return specs


def wan22_preprocess_command(job: dict[str, Any]) -> CommandSpec:
    vendor, models = _roots()
    inputs = job.get("inputs", {})
    prepared_dir = resolve_path(inputs["driving_video"]).parent
    save_path = prepared_dir / "wan22_process_results"
    return CommandSpec(
        phase="preprocess_wan22",
        cwd=str(vendor / "Wan2.2"),
        argv=[
            _backend_python("wan22"),
            str(vendor / "Wan2.2" / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"),
            "--ckpt_path",
            str(models / "Wan2.2-Animate-14B" / "process_checkpoint"),
            "--video_path",
            str(prepared_dir / "driving.mp4"),
            "--refer_path",
            str(resolve_path(inputs["reference_image"])),
            "--save_path",
            str(save_path),
            "--resolution_area",
            "1280",
            "720",
            "--retarget_flag",
            "--use_flux",
        ],
    )


def wan22_commands(job: dict[str, Any], smoke: bool = False) -> list[CommandSpec]:
    vendor, models = _roots()
    generation = job.get("generation", {})
    prepared_dir = resolve_path(job["inputs"]["driving_video"]).parent / "wan22_process_results"
    output_root = resolve_path(job.get("output", {}).get("root", "artifacts/pilot/runs/unknown"))
    seeds = list(generation.get("seeds", [1001]))[:1] if smoke else list(generation.get("seeds", [1001]))
    specs = []
    for seed in seeds:
        output = output_root / ("smoke" if smoke else "raw") / "wan22_animate" / f"seed_{seed}.mp4"
        specs.append(
            CommandSpec(
                phase="render_smoke" if smoke else "render",
                cwd=str(vendor / "Wan2.2"),
                output=str(output),
                argv=[
                    _backend_python("wan22"),
                    str(vendor / "Wan2.2" / "generate.py"),
                    "--task",
                    "animate-14B",
                    "--ckpt_dir",
                    str(models / "Wan2.2-Animate-14B"),
                    "--src_root_path",
                    str(prepared_dir),
                    "--refert_num",
                    "1",
                    "--base_seed",
                    str(seed),
                    "--save_file",
                    str(output),
                    "--offload_model",
                    "True",
                    "--convert_model_dtype",
                ],
            )
        )
    return specs


def wan22_s2v_commands(job: dict[str, Any], smoke: bool = False) -> list[CommandSpec]:
    """Build the official Wan2.2 Speech-to-Video CLI command.

    S2V is the actorless branch: reference image + speech audio, with an
    optional DWPose-style control video for the sceptre-hand gesture.
    """
    vendor, models = _roots()
    generation = job.get("generation", {})
    inputs = job.get("inputs", {})
    required = ("reference_image", "speech_audio")
    missing = [name for name in required if not inputs.get(name)]
    if missing:
        raise ConfigError("Wan2.2 S2V job is missing inputs: " + ", ".join(missing))
    output_root = resolve_path(job.get("output", {}).get("root", "artifacts/pilot/runs/unknown"))
    seeds = list(generation.get("seeds", [2101]))
    if smoke:
        seeds = seeds[:1]
    specs: list[CommandSpec] = []
    for seed in seeds:
        output = output_root / ("smoke" if smoke else "raw") / "wan22_s2v" / f"seed_{seed}.mp4"
        argv = [
            _backend_python("wan22"),
            str(vendor / "Wan2.2" / "generate.py"),
            "--task",
            "s2v-14B",
            "--size",
            str(generation.get("size", "720*1280")),
            "--ckpt_dir",
            str(models / "Wan2.2-S2V-14B"),
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--prompt",
            str(generation.get("prompt", "")),
            "--image",
            str(resolve_path(inputs["reference_image"])),
            "--audio",
            str(resolve_path(inputs["speech_audio"])),
            "--infer_frames",
            str(generation.get("infer_frames", 80)),
            "--sample_steps",
            str(generation.get("sample_steps", 40)),
            "--sample_shift",
            str(generation.get("sample_shift", 3.0)),
            "--sample_guide_scale",
            str(generation.get("guidance_scale", 4.5)),
            "--sample_solver",
            str(generation.get("solver", "unipc")),
            "--base_seed",
            str(seed),
            "--save_file",
            str(output),
        ]
        if inputs.get("pose_video"):
            argv.extend(["--pose_video", str(resolve_path(inputs["pose_video"]))])
        if generation.get("start_from_reference", True):
            argv.append("--start_from_ref")
        if smoke:
            argv.extend(["--num_clip", "1"])
        specs.append(
            CommandSpec(
                phase="render_smoke" if smoke else "render",
                cwd=str(vendor / "Wan2.2"),
                argv=argv,
                output=str(output),
            )
        )
    return specs


def wan_animate2_commands(job: dict[str, Any], smoke: bool = False) -> list[CommandSpec]:
    vendor, models = _roots()
    generation = job.get("generation", {})
    inputs = job.get("inputs", {})
    output_root = resolve_path(job.get("output", {}).get("root", "artifacts/pilot/runs/unknown"))
    output = output_root / ("smoke" if smoke else "raw") / "wan_animate2" / "distilled.mp4"
    argv = [
        _backend_python("wan_animate2"),
        str(vendor / "Wan-Animate-2" / "infer" / "wan_animate_2_demo.py"),
        "--prompt",
        str(generation.get("prompt", "")),
        "--refer-img-file",
        str(resolve_path(inputs["reference_image"])),
        "--refer-video-file",
        str(resolve_path(inputs.get("performance_video", inputs["driving_video"]))),
        "--config",
        str(vendor / "Wan-Animate-2" / "infer" / "wan_animate_2_distillation.yaml"),
        "--sample_guide_scale",
        "1.0",
        "--step",
        "10",
    ]
    return [CommandSpec(phase="render_frontier", cwd=str(vendor / "Wan-Animate-2" / "infer"), argv=argv, output=str(output))]


def render_commands(backend: str, job: dict[str, Any], smoke: bool = False) -> list[CommandSpec]:
    if backend == "scail2":
        return scail2_commands(job, smoke)
    if backend == "wan22_animate":
        return wan22_commands(job, smoke)
    if backend == "wan22_s2v":
        return wan22_s2v_commands(job, smoke)
    if backend == "wan_animate2":
        return wan_animate2_commands(job, smoke)
    raise ConfigError(f"No render adapter for backend: {backend}")


def rvc_command(voice_job: dict[str, Any]) -> CommandSpec:
    vendor, _ = _roots()
    conversion = voice_job.get("conversion", {})
    required = ("source_audio", "model", "index", "output")
    missing = [name for name in required if not conversion.get(name)]
    if missing:
        raise ConfigError("Voice config is missing: " + ", ".join(missing))
    output = resolve_path(conversion["output"])
    argv = [
        _backend_python("rvc"),
        "-m",
        "infer.cli",
        "--model",
        str(resolve_path(conversion["model"])),
        "--input",
        str(resolve_path(conversion["source_audio"])),
        "--output",
        str(output),
        "--index",
        str(resolve_path(conversion["index"])),
        "--pitch",
        str(conversion.get("pitch_semitones", 0)),
        "--f0-method",
        str(conversion.get("f0_method", "rmvpe")),
        "--index-rate",
        str(conversion.get("index_rate", 0.65)),
        "--resample-sr",
        str(conversion.get("resample_sr", 48000)),
        "--rms-mix-rate",
        str(conversion.get("rms_mix_rate", 0.85)),
        "--protect",
        str(conversion.get("protect", 0.33)),
        "--format",
        "wav",
    ]
    if conversion.get("overwrite") is True:
        argv.append("--overwrite")
    return CommandSpec(phase="voice_convert", cwd=str(vendor / "RVC"), argv=argv, output=str(output))


def preprocess_command(backend: str, job: dict[str, Any]) -> CommandSpec:
    if backend == "scail2":
        return scail2_preprocess_command(job)
    if backend == "wan22_animate":
        return wan22_preprocess_command(job)
    raise ConfigError(f"No preprocessing adapter for backend: {backend}")

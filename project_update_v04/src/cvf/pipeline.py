from __future__ import annotations

import copy
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .adapters import CommandSpec, preprocess_command, render_commands, rvc_command
from .config import atomic_dump_yaml, load_yaml, project_root, resolve_path
from .exceptions import AssetError, CommandError, ConfigError, GateError, RefuseOverwriteError
from .gates import (
    audit_registry,
    check_pronunciation,
    check_rights,
    check_runtime_lock,
    doctor,
)
from .hashing import file_sha256, tree_manifest
from .media import ffprobe, run_checked, video_summary


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _asset_is_required(asset: dict[str, Any], purpose: str) -> bool:
    required_for = asset.get("required_for", [])
    return purpose in required_for


def validate_assets(
    manifest_path: str | Path,
    purpose: str = "render",
    allow_missing: bool = False,
    validate_all: bool = False,
) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise ConfigError("Asset manifest has no assets mapping")

    results: dict[str, Any] = {}
    problems: list[str] = []
    hard_problems: list[str] = []
    for asset_id, asset in assets.items():
        if not isinstance(asset, dict):
            problems.append(f"{asset_id}: invalid record")
            continue
        required = _asset_is_required(asset, purpose)
        should_validate = required or validate_all
        raw_path = asset.get("path")
        if not raw_path:
            status = "missing_required" if required else "not_configured"
            results[asset_id] = {"status": status, "required": required}
            if required:
                problems.append(f"{asset_id}: required path is not configured")
            continue
        path = resolve_path(raw_path)
        if not path.is_file():
            results[asset_id] = {"status": "missing", "required": required, "path": str(path)}
            if required or validate_all:
                problems.append(f"{asset_id}: file does not exist")
            continue
        record: dict[str, Any] = {
            "status": "present_not_hashed" if not should_validate else "present",
            "required": required,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        if should_validate:
            actual = file_sha256(path)
            expected = str(asset.get("sha256") or "").upper()
            record["sha256"] = actual
            if expected and not expected.startswith("<") and actual != expected:
                record["status"] = "hash_mismatch"
                message = f"{asset_id}: SHA-256 mismatch"
                problems.append(message)
                hard_problems.append(message)
            else:
                record["status"] = "verified"
        results[asset_id] = record

    output = {"ok": not problems, "purpose": purpose, "assets": results, "problems": problems}
    if hard_problems or (problems and not allow_missing):
        raise AssetError("Asset validation failed: " + "; ".join(problems))
    return output


def lock_model(
    backend: str,
    weights_dir: str | Path,
    output_path: str | Path,
    registry_path: str | Path,
    license_ack: str | None,
) -> dict[str, Any]:
    if not license_ack or len(license_ack.strip()) < 4:
        raise GateError("--ack-license-record must identify the reviewed license/model-card record")
    registry = load_yaml(registry_path)
    model = registry.get("models", {}).get(backend)
    if not isinstance(model, dict):
        raise ConfigError(f"Unknown backend: {backend}")
    # SAM3 uses a gated dependency marker instead of a code commit in this registry.
    if backend != "sam3":
        audit_registry(registry_path, backend)
    root = resolve_path(weights_dir)
    manifest = tree_manifest(root)
    target = resolve_path(output_path)
    if target.exists():
        lock = load_yaml(target)
    else:
        lock = {"schema_version": 1, "models": {}}
    lock["generated_at"] = utc_now()
    lock.setdefault("models", {})[backend] = {
        "weights_state": "locked",
        "source_revision": model.get("source_revision"),
        "exact_checkpoint": model.get("exact_checkpoint"),
        "weights_dir": str(root),
        "license_record_acknowledged": license_ack.strip(),
        **manifest,
    }
    atomic_dump_yaml(lock, target)
    return {
        "ok": True,
        "backend": backend,
        "output": str(target),
        "tree_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def _job_and_backend(job_path: str | Path, backend_override: str | None = None) -> tuple[dict[str, Any], str]:
    job = load_yaml(job_path)
    backend = backend_override or job.get("backend")
    if not backend:
        raise ConfigError("Job has no backend")
    return job, str(backend)


def _job_input_status(job: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in job.get("inputs", {}).items():
        if not value:
            output[name] = {"exists": False, "path": None}
            continue
        path = resolve_path(value)
        output[name] = {"exists": path.is_file(), "path": str(path)}
    return output


def _required_render_approvals(job: dict[str, Any]) -> list[str]:
    configured = job.get("required_approvals")
    if isinstance(configured, list) and configured:
        return [str(name) for name in configured]
    if job.get("branch") == "fully_generative":
        return ["scene_plate", "gesture_storyboard", "converted_voice"]
    return ["scene_plate", "performance_p1"]


def _required_render_inputs(job: dict[str, Any], backend: str) -> list[str]:
    if backend == "wan22_s2v":
        required = ["reference_image", "speech_audio"]
        if job.get("generation", {}).get("pose_control_required", False):
            required.append("pose_video")
        return required
    return ["reference_image", "reference_mask", "driving_video", "driving_mask"]


def _check_render_approvals(job: dict[str, Any]) -> dict[str, Any]:
    approval = job.get("approval", {})
    required = _required_render_approvals(job)
    missing = [name for name in required if approval.get(name) is not True]
    if missing:
        raise GateError("Render approvals are missing: " + ", ".join(missing))
    return {"ok": True, "approved": required}


def render_plan(job_path: str | Path, backend_override: str | None = None, smoke: bool = False) -> dict[str, Any]:
    job, backend = _job_and_backend(job_path, backend_override)
    commands = render_commands(backend, job, smoke)
    gates: dict[str, Any] = {}
    for name, check in (
        ("rights", lambda: check_rights(job.get("rights_manifest", "configs/rights_manifest.local.yaml"))),
        ("pronunciation", check_pronunciation),
        ("registry", lambda: audit_registry(job.get("model_registry", "configs/model_registry.yaml"), backend)),
        ("runtime_lock", lambda: check_runtime_lock(backend, job)),
        ("human_approvals", lambda: _check_render_approvals(job)),
    ):
        try:
            gates[name] = {"ok": True, "detail": check()}
        except Exception as exc:  # planning reports gates but never spends or mutates
            gates[name] = {"ok": False, "error": str(exc)}
    input_status = _job_input_status(job)
    required_inputs = _required_render_inputs(job, backend)
    return {
        "ok_to_run": all(item["ok"] for item in gates.values()) and all(
            input_status.get(key, {}).get("exists") is True for key in required_inputs
        ),
        "job_id": job.get("job_id"),
        "backend": backend,
        "gpu_profile": job.get("gpu_profile"),
        "mode": "smoke" if smoke else "full",
        "external_uploads": False,
        "gates": gates,
        "inputs": input_status,
        "required_inputs": required_inputs,
        "commands": [spec.as_dict() for spec in commands],
        "expected_model_loads": [backend],
        "note": "Planning does not load weights or start paid inference.",
    }


def _find_asset(manifest: dict[str, Any], asset_id: str) -> Path:
    asset = manifest.get("assets", {}).get(asset_id)
    if not isinstance(asset, dict) or not asset.get("path"):
        raise AssetError(f"Asset is not configured: {asset_id}")
    path = resolve_path(asset["path"])
    if not path.is_file():
        raise AssetError(f"Asset does not exist: {asset_id} -> {path}")
    expected = str(asset.get("sha256") or "").upper()
    if expected and not expected.startswith("<") and file_sha256(path) != expected:
        raise AssetError(f"Asset hash mismatch: {asset_id}")
    return path


def prepare_performance(
    job_path: str | Path,
    assets_path: str | Path,
    profile: str | None = None,
    backend_override: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    job, backend = _job_and_backend(job_path, backend_override)
    profile_name = profile or str(job.get("gpu_profile"))
    assets = load_yaml(assets_path)
    source = _find_asset(assets, "performance_p1")
    reference = resolve_path(job["inputs"]["reference_image"])
    if not reference.is_file():
        raise AssetError(f"Reference image does not exist: {reference}")
    summary = video_summary(source)
    if summary["duration_seconds"] is None or not 3.0 <= summary["duration_seconds"] <= 20.0:
        raise AssetError("Performance duration must be between 3 and 20 seconds")
    if summary["height"] <= summary["width"]:
        raise AssetError("Performance must be vertical (height greater than width)")

    performance_target = resolve_path(job["inputs"]["performance_video"])
    prepared_dir = resolve_path(job["inputs"]["driving_video"]).parent
    normalized_command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-vf",
        "fps=25,scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "10",
        "-preset",
        "slow",
        str(performance_target),
    ]
    pre = preprocess_command(backend, job)
    plan = {
        "ok": True,
        "profile": profile_name,
        "source": summary,
        "normalize_command": normalized_command,
        "backend_preprocess": pre.as_dict(),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    check_rights(job.get("rights_manifest", "configs/rights_manifest.local.yaml"))
    check_pronunciation()
    doctor(profile_name)
    if backend == "scail2":
        check_runtime_lock("sam3", job)

    performance_target.parent.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    run_checked(normalized_command)
    shutil.copy2(performance_target, prepared_dir / "driving.mp4")
    shutil.copy2(reference, prepared_dir / "ref.png")

    for required in (Path(pre.cwd), Path(pre.argv[1])):
        if not required.exists():
            raise CommandError(f"Backend preprocessing installation is missing: {required}")
    run_checked(pre.argv, cwd=Path(pre.cwd), log_path=prepared_dir / "preprocess.log")
    expected = [
        prepared_dir / "ref_mask.jpg",
        prepared_dir / "rendered_v2.mp4",
        prepared_dir / "rendered_mask_v2.mp4",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise CommandError("Preprocessing completed without expected outputs: " + ", ".join(missing))
    plan["prepared_outputs"] = [str(path) for path in expected]
    return plan


def _make_scail_smoke_job(job: dict[str, Any]) -> dict[str, Any]:
    smoke_job = copy.deepcopy(job)
    prepared = resolve_path(job["inputs"]["driving_video"]).parent
    smoke_dir = prepared.parent / "smoke_prepared"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("reference_image", "ref.png"),
        ("reference_mask", "ref_mask.jpg"),
    ):
        source = resolve_path(job["inputs"][source_name])
        target = smoke_dir / target_name
        shutil.copy2(source, target)
        smoke_job["inputs"][source_name] = str(target)
    frame_count = int(job.get("generation", {}).get("window_frames", 77))
    for source_name, target_name in (
        ("driving_video", "rendered_v2.mp4"),
        ("driving_mask", "rendered_mask_v2.mp4"),
    ):
        source = resolve_path(job["inputs"][source_name])
        target = smoke_dir / target_name
        command = [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ]
        run_checked(command)
        smoke_job["inputs"][source_name] = str(target)
    return smoke_job


def run_render(
    job_path: str | Path,
    backend_override: str | None = None,
    profile: str | None = None,
    smoke: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    job, backend = _job_and_backend(job_path, backend_override)
    if dry_run:
        return render_plan(job_path, backend, smoke)
    profile_name = profile or str(job.get("gpu_profile"))
    check_rights(job.get("rights_manifest", "configs/rights_manifest.local.yaml"))
    check_pronunciation()
    audit_registry(job.get("model_registry", "configs/model_registry.yaml"), backend)
    check_runtime_lock(backend, job)
    _check_render_approvals(job)
    environment = doctor(profile_name)
    if not environment.get("inference_allowed"):
        raise GateError(f"Profile {profile_name} is preparation-only")

    required_names = _required_render_inputs(job, backend)
    missing = [
        name
        for name in required_names
        if not job.get("inputs", {}).get(name) or not resolve_path(job["inputs"][name]).is_file()
    ]
    if missing:
        raise AssetError("Prepared render inputs are missing: " + ", ".join(missing))
    execution_job = _make_scail_smoke_job(job) if smoke and backend == "scail2" else job
    specs = render_commands(backend, execution_job, smoke)
    for spec in specs:
        if spec.output and Path(spec.output).exists() and not force:
            raise RefuseOverwriteError(f"Output already exists; use --force: {spec.output}")

    run_id = f"{job.get('job_id')}_{backend}_{'smoke' if smoke else 'full'}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_root = resolve_path(job.get("output", {}).get("root", "artifacts/pilot/runs/unknown"))
    log_root = run_root / "logs" / run_id
    started = utc_now()
    completed_outputs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        cwd = Path(spec.cwd)
        if not cwd.is_dir() or not Path(spec.argv[1]).is_file():
            raise CommandError(f"Backend installation is missing for command {index}: {cwd}")
        if spec.output:
            Path(spec.output).parent.mkdir(parents=True, exist_ok=True)
        run_checked(spec.argv, cwd=cwd, log_path=log_root / f"{index:02d}_{spec.phase}.log")
        if spec.output:
            output = Path(spec.output)
            if not output.is_file():
                raise CommandError(f"Backend returned success but output is missing: {output}")
            completed_outputs.append(
                {"path": str(output), "bytes": output.stat().st_size, "sha256": file_sha256(output)}
            )

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "job_id": job.get("job_id"),
        "backend": backend,
        "mode": "smoke" if smoke else "full",
        "status": "completed",
        "started_at": started,
        "finished_at": utc_now(),
        "profile": profile_name,
        "gpu": environment.get("gpus"),
        "commands": [spec.as_dict() for spec in specs],
        "outputs": completed_outputs,
    }
    manifest_path = log_root / "result_manifest.yaml"
    atomic_dump_yaml(result, manifest_path)
    result["manifest"] = str(manifest_path)
    return result


def voice_plan(config_path: str | Path) -> dict[str, Any]:
    voice_job = load_yaml(config_path)
    command = rvc_command(voice_job)
    conversion = voice_job.get("conversion", {})
    paths = {
        name: {
            "path": str(resolve_path(conversion[name])),
            "exists": resolve_path(conversion[name]).is_file(),
        }
        for name in ("source_audio", "model", "index")
        if conversion.get(name)
    }
    gates: dict[str, Any] = {}
    for name, check in (
        ("rights", lambda: check_rights(voice_job.get("rights_manifest", "configs/rights_manifest.local.yaml"))),
        ("pronunciation", check_pronunciation),
        ("registry", lambda: audit_registry(voice_job.get("model_registry", "configs/model_registry.yaml"), "rvc")),
        ("runtime_lock", lambda: check_runtime_lock("rvc", voice_job)),
    ):
        try:
            gates[name] = {"ok": True, "detail": check()}
        except Exception as exc:
            gates[name] = {"ok": False, "error": str(exc)}
    approval = voice_job.get("approval", {})
    gates["checkpoint_abx"] = {
        "ok": approval.get("checkpoint_abx") is True,
        "error": None if approval.get("checkpoint_abx") is True else "RVC checkpoint ABX is not approved",
    }
    return {
        "ok_to_run": all(item["ok"] for item in gates.values()) and all(item["exists"] for item in paths.values()),
        "voice_job_id": voice_job.get("voice_job_id"),
        "backend": "rvc",
        "gpu_profile": voice_job.get("gpu_profile"),
        "gates": gates,
        "inputs": paths,
        "command": command.as_dict(),
        "note": "The source timing is preserved; final text/voice still require ASR and human approval.",
    }


def _duration_seconds(path: Path) -> float:
    probe = ffprobe(path)
    value = probe.get("format", {}).get("duration")
    if value is None:
        raise AssetError(f"Media duration is unavailable: {path}")
    return float(value)


def run_voice_convert(
    config_path: str | Path,
    profile: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return voice_plan(config_path)
    voice_job = load_yaml(config_path)
    check_rights(voice_job.get("rights_manifest", "configs/rights_manifest.local.yaml"))
    check_pronunciation()
    audit_registry(voice_job.get("model_registry", "configs/model_registry.yaml"), "rvc")
    check_runtime_lock("rvc", voice_job)
    profile_name = profile or str(voice_job.get("gpu_profile"))
    environment = doctor(profile_name)
    if not environment.get("inference_allowed"):
        raise GateError(f"Profile {profile_name} is preparation-only")
    if voice_job.get("approval", {}).get("checkpoint_abx") is not True:
        raise GateError("RVC checkpoint must pass ABX approval before conversion")
    spec = rvc_command(voice_job)
    input_path = resolve_path(voice_job["conversion"]["source_audio"])
    model_path = resolve_path(voice_job["conversion"]["model"])
    index_path = resolve_path(voice_job["conversion"]["index"])
    for name, path in (("source_audio", input_path), ("model", model_path), ("index", index_path)):
        if not path.is_file():
            raise AssetError(f"Voice {name} does not exist: {path}")
    output = Path(spec.output or "")
    if output.exists() and not force:
        raise RefuseOverwriteError(f"Voice output already exists; use --force: {output}")
    if force and "--overwrite" not in spec.argv:
        spec.argv.append("--overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(spec.cwd)
    cli_module = repo / "infer" / "cli.py"
    if not repo.is_dir() or not cli_module.is_file() or not Path(spec.argv[0]).is_file():
        raise CommandError(f"RVC installation is incomplete: {repo}")
    run_checked(spec.argv, cwd=repo, log_path=output.parent / "rvc_convert.log")
    if not output.is_file():
        raise CommandError(f"RVC returned success but output is missing: {output}")
    input_duration = _duration_seconds(input_path)
    output_duration = _duration_seconds(output)
    delta_ms = abs(output_duration - input_duration) * 1000
    if delta_ms > 80:
        raise CommandError(f"RVC timing drift is {delta_ms:.1f} ms, limit is 80 ms")
    result = {
        "schema_version": 1,
        "voice_job_id": voice_job.get("voice_job_id"),
        "status": "completed_pending_human_approval",
        "input_duration_seconds": input_duration,
        "output_duration_seconds": output_duration,
        "timing_delta_ms": round(delta_ms, 3),
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": file_sha256(output)},
        "command": spec.as_dict(),
    }
    atomic_dump_yaml(result, output.parent / "voice_result_manifest.yaml")
    return result


def _post_commands(post_job: dict[str, Any]) -> list[CommandSpec]:
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    inputs = post_job.get("inputs", {})
    outputs = post_job.get("outputs", {})
    video_cfg = post_job.get("video", {})
    audio_cfg = post_job.get("audio", {})
    source_video = resolve_path(inputs.get("accepted_video", ""))
    source_audio = resolve_path(inputs.get("approved_voice", ""))
    master = resolve_path(outputs.get("master", ""))
    delivery = resolve_path(outputs.get("delivery", ""))
    width = int(video_cfg.get("width", 1080))
    height = int(video_cfg.get("height", 1920))
    fps = int(video_cfg.get("fps", 25))
    vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    master_command = CommandSpec(
        phase="post_master",
        cwd=str(project_root()),
        output=str(master),
        argv=[
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(source_video),
            "-i",
            str(source_audio),
            "-map_metadata",
            "-1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            vf,
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            "-c:a",
            "pcm_s24le",
            "-ar",
            str(audio_cfg.get("sample_rate", 48000)),
            "-shortest",
            str(master),
        ],
    )
    delivery_command = CommandSpec(
        phase="post_delivery",
        cwd=str(project_root()),
        output=str(delivery),
        argv=[
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(master),
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-crf",
            str(video_cfg.get("delivery_crf", 16)),
            "-preset",
            str(video_cfg.get("delivery_preset", "slow")),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            str(audio_cfg.get("delivery_bitrate", "320k")),
            "-ar",
            str(audio_cfg.get("sample_rate", 48000)),
            "-movflags",
            "+faststart",
            str(delivery),
        ],
    )
    return [master_command, delivery_command]


def post_plan(config_path: str | Path) -> dict[str, Any]:
    job = load_yaml(config_path)
    commands = _post_commands(job)
    input_status = {
        key: {"path": str(resolve_path(value)), "exists": resolve_path(value).is_file()}
        for key, value in job.get("inputs", {}).items()
    }
    gates: dict[str, Any] = {}
    for name, check in (
        ("rights", lambda: check_rights(job.get("rights_manifest", "configs/rights_manifest.local.yaml"))),
        ("pronunciation", check_pronunciation),
    ):
        try:
            gates[name] = {"ok": True, "detail": check()}
        except Exception as exc:
            gates[name] = {"ok": False, "error": str(exc)}
    approvals = job.get("approval", {})
    missing_approvals = [
        name for name in ("scene_plate", "render_candidate", "converted_voice")
        if approvals.get(name) is not True
    ]
    gates["human_approvals"] = {
        "ok": not missing_approvals,
        "error": None if not missing_approvals else "Missing: " + ", ".join(missing_approvals),
    }
    return {
        "ok_to_run": all(item["ok"] for item in gates.values()) and all(item["exists"] for item in input_status.values()),
        "post_job_id": job.get("post_job_id"),
        "gpu_profile": job.get("gpu_profile"),
        "gates": gates,
        "inputs": input_status,
        "commands": [command.as_dict() for command in commands],
    }


def run_post_assemble(
    config_path: str | Path,
    profile: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return post_plan(config_path)
    job = load_yaml(config_path)
    check_rights(job.get("rights_manifest", "configs/rights_manifest.local.yaml"))
    check_pronunciation()
    approvals = job.get("approval", {})
    missing_approvals = [
        name for name in ("scene_plate", "render_candidate", "converted_voice")
        if approvals.get(name) is not True
    ]
    if missing_approvals:
        raise GateError("Post approvals are missing: " + ", ".join(missing_approvals))
    profile_name = profile or str(job.get("gpu_profile"))
    environment = doctor(profile_name)
    if not environment.get("inference_allowed"):
        raise GateError(f"Profile {profile_name} is preparation-only; final assembly is assigned to the rented host")
    for name, value in job.get("inputs", {}).items():
        path = resolve_path(value)
        if not path.is_file():
            raise AssetError(f"Post input does not exist: {name} -> {path}")
    commands = _post_commands(job)
    for command in commands:
        output = Path(command.output or "")
        if output.exists() and not force:
            raise RefuseOverwriteError(f"Post output already exists; use --force: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        run_checked(command.argv, cwd=Path(command.cwd), log_path=output.parent / f"{command.phase}.log")
        if not output.is_file():
            raise CommandError(f"Post command returned success but output is missing: {output}")
    master = Path(commands[0].output or "")
    delivery = Path(commands[1].output or "")
    master_summary = video_summary(master)
    delivery_summary = video_summary(delivery)
    expected = job.get("video", {})
    problems = []
    if delivery_summary["width"] != int(expected.get("width", 1080)):
        problems.append("delivery width mismatch")
    if delivery_summary["height"] != int(expected.get("height", 1920)):
        problems.append("delivery height mismatch")
    if delivery_summary["duration_seconds"] and delivery_summary["duration_seconds"] > 20.0:
        problems.append("delivery exceeds 20 seconds")
    if delivery_summary["audio_codec"] is None:
        problems.append("delivery has no audio stream")
    if problems:
        raise CommandError("Post QC failed: " + "; ".join(problems))
    result = {
        "schema_version": 1,
        "post_job_id": job.get("post_job_id"),
        "status": "completed_pending_content_qc",
        "master": {**master_summary, "sha256": file_sha256(master)},
        "delivery": {**delivery_summary, "sha256": file_sha256(delivery)},
        "commands": [command.as_dict() for command in commands],
    }
    atomic_dump_yaml(result, delivery.parent / "post_result_manifest.yaml")
    return result


def _bundle_project_paths(root: Path, output: Path) -> list[Path]:
    allowed_files = {
        "AGENTS.md",
        "PLANS.md",
        "README.md",
        "SPEC.md",
        "DECISIONS.md",
        "CHANGELOG.md",
        "LICENSES.md",
        "SECURITY.md",
        "EVIDENCE_POLICY.md",
        "SHOOTING_GUIDE_RU.md",
        "SERVER_RENTAL_CHECKLIST_RU.md",
        "ACTORLESS_PIPELINE_RU.md",
        "pyproject.toml",
        "Makefile",
        ".env.example",
        ".gitignore",
        ".gitattributes",
    }
    allowed_dirs = {"configs", "evidence", "schemas", "src", "scripts", "docker", "tests", "tools", "reports"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root)
        if relative.name in {"assets.local.yaml", "assets.actorless.local.yaml"}:
            continue
        if (
            "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or any(part.endswith(".egg-info") for part in relative.parts)
        ):
            continue
        if relative.parts[0] in allowed_dirs or relative.as_posix() in allowed_files:
            paths.append(path)
        elif relative.parts[:3] == ("artifacts", "plates", "candidates") and path.name in {
            "plate_throne_clean_landscape_v03_aligned_source_locked.png",
            "plate_throne_clean_vertical_v04_camera_facing.png",
        }:
            paths.append(path)
        elif relative.parts[:3] == ("artifacts", "storyboards", "actorless_gesture_v02_camera_facing") and (
            path.name == "contact_sheet_15.png"
            or path.name in {f"frame_{index:02d}.png" for index in range(1, 16)}
        ):
            paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def build_remote_bundle(
    job_path: str | Path,
    assets_path: str | Path,
    output_path: str | Path,
    allow_missing: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    root = project_root()
    output = resolve_path(output_path)
    if output.exists() and not force:
        raise RefuseOverwriteError(f"Bundle already exists; use --force: {output}")
    job = load_yaml(job_path)
    manifest = load_yaml(assets_path)
    remote_assets_name = resolve_path(assets_path).name
    assets = manifest.get("assets", {})
    missing: list[str] = []
    transferred: list[dict[str, Any]] = []
    remote_assets: dict[str, Any] = {}
    for asset_id, asset in assets.items():
        if not isinstance(asset, dict) or asset.get("transfer") is not True:
            continue
        raw_path = asset.get("path")
        required = "render" in asset.get("required_for", [])
        if not raw_path:
            if required:
                missing.append(asset_id)
                remote_assets[asset_id] = {
                    "path": None,
                    "sha256": None,
                    "transfer": True,
                    "required_for": asset.get("required_for", []),
                }
            continue
        path = resolve_path(raw_path)
        if not path.is_file():
            if required:
                missing.append(asset_id)
                remote_assets[asset_id] = {
                    "path": None,
                    "sha256": None,
                    "transfer": True,
                    "required_for": asset.get("required_for", []),
                }
            continue
        actual = file_sha256(path)
        expected = str(asset.get("sha256") or "").upper()
        if expected and not expected.startswith("<") and actual != expected:
            raise AssetError(f"Cannot bundle {asset_id}: SHA-256 mismatch")
        try:
            archive_rel = path.relative_to(root)
        except ValueError:
            # External Windows files may have Cyrillic or otherwise host-specific
            # names. Give them a deterministic ASCII name inside the Linux bundle.
            archive_rel = Path("data") / "bundle_inputs" / asset_id / f"{asset_id}{path.suffix.lower()}"
        transferred.append(
            {"asset_id": asset_id, "source": path, "archive_rel": archive_rel, "bytes": path.stat().st_size, "sha256": actual}
        )
        remote_assets[asset_id] = {
            "path": archive_rel.as_posix(),
            "sha256": actual,
            "transfer": True,
            "required_for": asset.get("required_for", []),
        }
    if missing and not allow_missing:
        raise AssetError("Cannot build runnable bundle; missing transfer assets: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = root.name
    project_paths = _bundle_project_paths(root, output)
    project_path_set = set(project_paths)
    with tarfile.open(output, "w:gz") as archive:
        for path in project_paths:
            relative = path.relative_to(root)
            archive.add(path, arcname=(Path(prefix) / relative).as_posix(), recursive=False)
        for item in transferred:
            if item["source"] not in project_path_set:
                archive.add(item["source"], arcname=(Path(prefix) / item["archive_rel"]).as_posix(), recursive=False)
        remote_manifest = {"schema_version": 1, "assets": remote_assets}
        payload = yaml.safe_dump(remote_manifest, allow_unicode=True, sort_keys=False).encode("utf-8")
        info = tarfile.TarInfo((Path(prefix) / "configs" / remote_assets_name).as_posix())
        info.size = len(payload)
        info.mtime = int(dt.datetime.now().timestamp())
        archive.addfile(info, io.BytesIO(payload))
        bundle_manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "job_id": job.get("job_id"),
            "missing_allowed": missing,
            "assets": [
                {
                    key: (value.as_posix() if isinstance(value, Path) else value)
                    for key, value in item.items()
                    if key != "source"
                }
                for item in transferred
            ],
        }
        bundle_payload = json.dumps(bundle_manifest, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        bundle_info = tarfile.TarInfo((Path(prefix) / "artifacts" / "provenance" / "bundle_manifest.json").as_posix())
        bundle_info.size = len(bundle_payload)
        bundle_info.mtime = info.mtime
        archive.addfile(bundle_info, io.BytesIO(bundle_payload))

    return {
        "ok": not missing,
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "asset_count": len(transferred),
        "missing": missing,
    }

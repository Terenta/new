from __future__ import annotations

from cvf.adapters import render_commands, rvc_command


def test_scail_command_uses_official_long_video_flags() -> None:
    job = {
        "inputs": {
            "reference_image": "ref.png",
            "reference_mask": "ref_mask.jpg",
            "driving_video": "rendered_v2.mp4",
            "driving_mask": "rendered_mask_v2.mp4",
        },
        "generation": {
            "width": 704,
            "height": 1280,
            "window_frames": 77,
            "overlap_frames": 8,
            "seeds": [1001],
            "prompt": "A seated tsar speaks.",
        },
        "output": {"root": "artifacts/test"},
    }
    argv = render_commands("scail2", job)[0].argv
    assert argv[argv.index("--model") + 1] == "SCAIL-14B"
    assert argv[argv.index("--target_w") + 1] == "704"
    assert argv[argv.index("--target_h") + 1] == "1280"
    assert argv[argv.index("--segment_len") + 1] == "77"
    assert argv[argv.index("--segment_overlap") + 1] == "8"
    assert "--save_file" in argv


def test_smoke_uses_one_seed() -> None:
    job = {
        "inputs": {
            "reference_image": "ref.png",
            "reference_mask": "ref_mask.jpg",
            "driving_video": "rendered_v2.mp4",
            "driving_mask": "rendered_mask_v2.mp4",
        },
        "generation": {"seeds": [1, 2, 3]},
        "output": {"root": "artifacts/test"},
    }
    assert len(render_commands("scail2", job, smoke=True)) == 1


def test_actorless_s2v_uses_audio_pose_and_reference_start() -> None:
    job = {
        "inputs": {
            "reference_image": "ref.png",
            "speech_audio": "speech.wav",
            "pose_video": "pose.mp4",
        },
        "generation": {
            "size": "720*1280",
            "infer_frames": 80,
            "seeds": [2101, 2102],
            "prompt": "A seated tsar speaks and raises one ceremonial staff.",
            "start_from_reference": True,
        },
        "output": {"root": "artifacts/test"},
    }
    specs = render_commands("wan22_s2v", job, smoke=True)
    assert len(specs) == 1
    argv = specs[0].argv
    assert argv[argv.index("--task") + 1] == "s2v-14B"
    assert argv[argv.index("--size") + 1] == "720*1280"
    assert argv[argv.index("--infer_frames") + 1] == "80"
    assert "--audio" in argv
    assert "--pose_video" in argv
    assert "--start_from_ref" in argv
    assert argv[argv.index("--num_clip") + 1] == "1"


def test_rvc_command_preserves_timing_parameters() -> None:
    job = {
        "conversion": {
            "source_audio": "input.wav",
            "model": "model.pth",
            "index": "model.index",
            "output": "output.wav",
            "pitch_semitones": 0,
            "index_rate": 0.65,
            "resample_sr": 48000,
            "rms_mix_rate": 0.85,
            "protect": 0.33,
        }
    }
    argv = rvc_command(job).argv
    assert argv[1:3] == ["-m", "infer.cli"]
    assert argv[argv.index("--pitch") + 1] == "0"
    assert argv[argv.index("--resample-sr") + 1] == "48000"
    assert argv[argv.index("--index-rate") + 1] == "0.65"

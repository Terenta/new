from __future__ import annotations

from cvf.cli import main


def test_render_plan_never_starts_inference(capsys) -> None:
    code = main(["render", "plan", "--job", "configs/jobs/ivan_throne_motion.yaml", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert '"ok_to_run": false' in captured.out
    assert "Planning does not load weights" in captured.out


def test_rights_cli_fails_closed(capsys) -> None:
    code = main(
        ["rights", "check", "--manifest", "configs/rights_manifest.local.example.yaml", "--json"]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "Rights gate failed" in captured.err


def test_voice_and_post_plans_are_non_mutating(capsys) -> None:
    assert main(["voice", "plan", "--config", "configs/voice/ivan_rvc.yaml", "--json"]) == 0
    voice_output = capsys.readouterr().out
    assert '"ok_to_run": false' in voice_output
    assert main(["post", "plan", "--config", "configs/post.yaml", "--json"]) == 0
    post_output = capsys.readouterr().out
    assert '"ok_to_run": false' in post_output

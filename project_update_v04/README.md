# Ivan pilot v04 voice curation update

This code-only overlay records the reviewed 15-slice Ivan Grozny voice set,
creates a separate immutable curated RVC experiment, and makes that experiment
the default training target. No source voice, film or performance media is
included in this update.

Apply from the repository root on the server:

```bash
(cd project_update_v04 && sha256sum -c SHA256SUMS)
rsync -a project_update_v04/ /workspace/character-video-factory-ivan-pilot/
chmod +x /workspace/character-video-factory-ivan-pilot/scripts/apply_voice_curation.sh
```

`SHA256SUMS` intentionally omits this README and covers every overlay payload.

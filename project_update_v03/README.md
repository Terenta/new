# Ivan pilot v03 server update

This overlay replaces the one-arm gesture with a synchronized two-arm gesture,
records the user's rights assertion, and contains the verified Blackwell/RVC
bootstrap fixes discovered during deployment.

Apply from the repository root on the server:

```bash
sha256sum -c project_update_v03/SHA256SUMS
rsync -a project_update_v03/ /workspace/character-video-factory-ivan-pilot/
```

`SHA256SUMS` intentionally omits this README and covers every overlay payload.

# Ivan pilot server handoff

Reassemble and verify on Linux:

```bash
cat bundle/ivan-pilot-actorless-prep.tar.gz.part-* > ivan-pilot-actorless-prep.tar.gz
sha256sum -c bundle/ivan-pilot-actorless-prep.tar.gz.sha256
tar -xzf ivan-pilot-actorless-prep.tar.gz
```

Expected archive SHA-256:

`94C4CBC045BDBBD76731EF48920DC073D535B9F596CE985A6A8CBB96FC89C591`

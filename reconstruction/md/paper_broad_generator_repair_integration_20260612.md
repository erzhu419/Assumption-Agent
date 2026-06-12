# Paper Broad Generator Repair Integration

- pass: `True`
- original trigger utility: `0.4161` CI `[0.3698, 0.4619]`
- v1 repair trigger utility: `0.3183` CI `[0.2792, 0.36]`
- v2 repair trigger utility: `0.5462` CI `[0.4985, 0.5956]`
- v2 fresh calls: `720/720`
- v2 selected candidates: `8` (original `80`)
- v2 control loss: `0.1456` CI `[0.0922, 0.2022]`

## Interpretation

The raw broad generator failed.  The repaired generator uses fresh failure evidence as a selector and abstains from low-support families.  The resulting qualified frontier keeps the 720-call budget and passes the all-candidate trigger gate on a new live rerun.

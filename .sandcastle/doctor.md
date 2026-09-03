# Harness Doctor

You are checking whether this repository is ready for reliable preflight and Night Shift work.

Read:

1. `AGENTS.md`
2. `AGENT_LOOP.md`
3. `~/repositories/.night-shift/HARNESS_PRINCIPLES.md` if available
4. `~/repositories/.night-shift/SANDCASTLE.md` if available
5. `~/repositories/.night-shift/SPEC_PREFLIGHT.md` if available
6. repository metadata such as `package.json`, `pyproject.toml`, `README.md`, `.env.example`, and CI files when present
7. `.sandcastle/sandbox.json`

Do not implement product features.

## Checks

Verify whether these exist and are useful:

- `AGENTS.md`
- `AGENT_LOOP.md`
- `.sandcastle/implement-night-shift.md`
- `.sandcastle/preflight-specs.md`
- `.sandcastle/sandbox.json`
- `Specs/README.md`
- `docs/README.md`
- `docs/testing.md`
- `docs/architecture.md`
- `docs/domain.md`
- `docs/style-guide.md`
- `docs/common-pitfalls.md`
- `TODO.md`
- `CHANGELOG.md`
- environment example file if the repo needs env vars
- correct sandbox profile type: `generic`, `compose`, or `custom-image`
- service dependencies documented in the sandbox profile
- exact test, lint, typecheck, and build commands
- clear local setup instructions
- ready specs and draft specs are distinguishable
- prior `Specs/preflight-report.md` if present

## Output

Write or update:

```text
docs/harness-doctor.md
```

The report must include:

- overall readiness score from 0 to 5
- missing files
- stale or placeholder files
- missing commands
- missing environment/setup notes
- spec readiness concerns
- recommended next edits
- suggested mechanical checks to add

You may make small documentation-only edits that improve routing, such as creating missing placeholder docs from the expected structure. Do not change product code.

When complete, output:

```text
<promise>DOCTOR_COMPLETE</promise>
```

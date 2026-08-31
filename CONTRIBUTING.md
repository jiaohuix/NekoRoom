# Contributing to NekoRoom

Thanks for your interest in NekoRoom.

NekoRoom is currently an early-stage project. The architecture is evolving quickly, so focused pull requests are preferred over large unrelated changes.

## Development workflow

NekoRoom uses a **fork + pull request** workflow.

### 1. Fork the repository

Fork:

```text
NekoAI-Labs/NekoRoom
```

to your own GitHub account.

### 2. Configure remotes

Clone your fork:

```bash
git clone git@github.com:<your-account>/NekoRoom.git
cd NekoRoom
```

Add the official repository as `upstream`:

```bash
git remote add upstream git@github.com:NekoAI-Labs/NekoRoom.git
git remote -v
```

Recommended convention:

```text
origin   = your fork
upstream = NekoAI-Labs/NekoRoom
```

### 3. Sync before starting work

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main
```

### 4. Create a focused branch

Examples:

```bash
git checkout -b feat/agent-runtime
git checkout -b feat/voice-provider
git checkout -b fix/session-restore
git checkout -b refactor/nekochat-to-nekoroom
git checkout -b docs/update-readme
```

Recommended prefixes:

- `feat/` — new functionality
- `fix/` — bug fixes
- `refactor/` — structural changes without intended behavior changes
- `docs/` — documentation
- `test/` — tests
- `chore/` — tooling, dependencies, repository maintenance

### 5. Commit clearly

Suggested Conventional Commit style:

```text
feat: add local model provider
fix: restore conversation state after restart
refactor: isolate agent session state
docs: document voice provider setup
test: add memory retrieval regression tests
chore: update development dependencies
```

### 6. Push and open a pull request

```bash
git push -u origin <your-branch>
```

Then open a pull request against:

```text
NekoAI-Labs/NekoRoom:main
```

## Initial NekoChat import

The first public source release is expected to contain existing **NekoChat** names internally.

Please do not perform an unrelated mass rename in the initial source import.

The NekoChat → NekoRoom migration should happen through dedicated follow-up pull requests so functional changes and naming changes remain reviewable.

## Pull request expectations

Before opening a PR:

- keep the change focused;
- run relevant tests when available;
- review `git diff` yourself;
- do not commit API keys, tokens, credentials, private URLs, personal data, local databases, logs, or model secrets;
- do not add model weights or datasets unless their redistribution rights are clear;
- identify third-party code or assets and preserve required license notices;
- update documentation when behavior or configuration changes.

## Licensing of contributions

Unless explicitly agreed otherwise in writing, contributions submitted to this repository are expected to be provided under the repository's applicable open-source license, currently **AGPL-3.0-only** for the main source code.

Do not submit code, datasets, models, media, or other material that you do not have the right to contribute.

The project may introduce a Contributor License Agreement (CLA) before accepting substantial third-party contributions in order to keep long-term project licensing and commercial options manageable.

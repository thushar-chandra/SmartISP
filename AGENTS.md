# AGENTS.md

Status: placeholder. This repository is currently empty (no source, manifests, config, or git history as of 2026-11-06). Replace this file with real guidance once code exists.

## Environment

- Workspace lives inside WSL2: `/home/thushar/projects/SmartISP`, opened from Windows via the UNC path `\\wsl.localhost\Ubuntu\home\thushar\projects\SmartISP`.
- The shell is Windows PowerShell 5.1. For Linux tooling, wrap commands: `wsl -d Ubuntu -- bash -lc "cd /home/thushar/projects/SmartISP && <command>"`.
- Not a git repository yet; run `git init` before relying on git-based workflows.

## When populating this file

Read the repo's executable sources first (`README*`, package/build manifests, lockfiles, CI workflows, lint/test/typecheck config) and document only repo-specific facts: exact commands, single-test invocation, command ordering, package boundaries, codegen/migrations, env loading, and non-default conventions. Verify every claim against config or scripts.

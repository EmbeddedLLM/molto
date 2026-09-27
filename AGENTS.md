# Guide for AI agents

This `AGENTS.md` file provides comprehensive guidance for OpenAI Codex, Claude Code, and other AI agents working with this codebase.

When working on the Python Molto library, read `python/ARCHITECTURE.md` and `python/CONVENTIONS.md` before making changes.

The user's job as a senior software engineer is dependent on you implementing clean code with elegant architecture.

## Core Principles

- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Simplicity First**: Make every change as simple as possible. Only touch what's necessary. Impact minimal code. Over-complication leads to bugs.
  1. Always start with thorough understanding of the problem and motivation.
  2. Does this functionality need to exist? Challenge or skip it when it does not (YAGNI).
  3. Already in this codebase? Reuse or extend it; do not rewrite it.
  4. Does the standard library do it? Use it.
  5. Does the platform do it? Use it.
  6. Does an installed dependency do it? Use it.
  7. Can it be one line? Make it one line.
  8. Only then, implement the minimum that works.
- **Simplify After Implementation**: Treat the first working version and the requirements behind it as drafts.
  1. Run relevant tests to uncover hidden requirements and assumptions.
  2. Relate every code path to a concrete requirement and challenge whether that requirement is necessary.
  3. Review the full diff for unnecessary abstractions, branches, defensive code, and overly strict guarantees.
  4. Prefer sufficient correctness over absolute correctness when the trade-off is appropriate, such as eventual consistency or simple retries for cheap, recoverable failures.
  5. Simplify immediately when behavior is preserved; otherwise present the proposed change, benefits, and trade-offs to the user.
- **Protect Secrets in Command Output**: Never print resolved configuration, environment variables, credentials, or command output that may expose secrets. When validating Compose files that use environment files, use `docker compose config --quiet`; do not print fully resolved Compose configuration because it can expand secret environment values into command output.

## Workflow Orchestration

### 1. Planning

- For non-trivial work, create a concise implementation plan before editing.
- Proceed without waiting for approval when the requested outcome is clear and the changes are low-risk.
- Ask for clarification only when ambiguity would materially change behavior, public interfaces, data, security, or architecture.
- Re-plan when evidence invalidates the current approach.
- Keep planning proportional to the task; do not create process overhead for straightforward changes.

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Task Tracking

- Use `tasks/todo.md` for substantial, multi-stage work when persistent tracking is useful.
- Do not create or modify `tasks/todo.md` for small or routine changes; the completion record below is still required.
- At the end of every feature, bug fix, or task, before the final handoff, create `tasks/YYYY-MM-DD-<task-name>.md`. Use the completion date and a short, descriptive, lowercase task name with hyphens.
- Write the completion record using the structure and wording rules in [PR Descriptions](#pr-descriptions).
- Record only durable, project-specific lessons in `tasks/lessons.md`.
- Do not record one-off user preferences, speculative conclusions, or conversational details.
- A user correction is evidence to reconsider the approach, not an automatic reason to create a permanent rule.

### 4. Verification Before Done

- Run the strongest applicable verification available in the current environment.
- Never claim that a check passed unless it was actually run and passed.
- When a check cannot be run, report the exact command not run and the reason.
- Distinguish failures introduced by the current change from pre-existing or environment-dependent failures.
- Verify the behavior affected by the change, not merely that the code compiles.
- All tests must have docstrings that state their intention, goal, purpose, or the regression they guard against. Use phrasing such as "Ensures ..." or "Asserts that ..." to describe the behavior being protected.
- Add or update regression tests when fixing a reproducible defect. For any public provider behavior with an OpenAI SDK-equivalent contract, include cassette-backed parity coverage that compares Molto against parsed OpenAI SDK responses or events; IR and provider-setting unit tests may supplement this coverage, but must not replace it.
- Read `tests/cassettes/README.md` before recording, editing, or reviewing cassette fixtures.
- Never run tests in cassette-recording mode. Test runs must replay them only.
- If any of the committed cassettes are demonstrably wrong, you may update them, but you must explicitly mention the changes to the user in your final review or reply.
- Never update uncommitted cassettes, always commit them first, and use their diff to guide your changes.
- Compare behavior before and after the change when practical.
- Run the applicable checks defined in the Linting Workflow.
- Always check whether any project documentation needs updating, even when documentation was not mentioned in the task. Update all affected docs before reporting completion.
- Keep verification proportional to the scope and risk of the change. For behavior changes, run focused relevant tests; run `pytest` for broad changes or when the full offline replay suite is required.

#### Python Tests

- Run tests (especially async tests that invoke the OpenAI or Anthropic SDK) with escalated permissions on the first attempt. Their first-request platform detection uses `asyncio.to_thread()`, while the sandbox blocks the Unix socketpair wakeup that asyncio uses to deliver the worker result. Inside the sandbox these tests can therefore stall before reaching `MockTransport`; this is an environment limitation, not a cassette or SDK parsing failure.
- Resolve the interpreter before blaming a dependency for a version-specific failure. A bare `--python 3.14` request selects the newest build the installed uv knows about, and an older uv resolved that to a pre-release (`cpython-3.14.0a5`, under `~/.local/share/uv/python/`). Print `python -VV` in the failing environment and retry with an up-to-date uv before concluding anything about the code.

#### Stalled Command Cleanup

- Do not launch verification commands with shell backgrounding such as `&`. Keep each command attached to a tracked terminal or execution session.
- Use bounded waits and provide a progress update at least every 60 seconds while a long-running command is active.
- If a command stops making progress, interrupt or terminate its exact tracked session. Do not use a broad process kill that could stop unrelated user work.
- Confirm that the command exited, then check for orphaned test processes with narrowly scoped process queries. For Python tests, check both `pgrep -ax pytest` and `pgrep -af '^/absolute/workspace/path/.venv/bin/python'`, substituting the current workspace path. An empty result from both confirms that no repository pytest or virtual-environment Python process remains.
- If either query finds a process started by the interrupted command, terminate that exact PID or tracked session and repeat both checks. Never leave a verification process running after handing control back to the user.
- Report the exact interrupted command, where it stalled or timed out, how it was stopped, and the cleanup result. An interrupted, timed-out, or incomplete command is not a passing check.

Before reporting completion, review the final diff and ask:

- Does every change relate to a requirement?
- Did the change introduce unnecessary complexity or scope?
- Would a staff engineer approve the implementation and verification evidence?

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Create the dated completion record in `tasks/` using the required [PR description format](#pr-descriptions).
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

### 8. Linting Workflow

- Run formatting, linting, and static analysis after implementation and before reporting completion.
- The `.github/workflows/lint.yml` workflow is the source of truth. Before running checks, inspect it and use the commands, installation method, configuration, and tool versions selected by that workflow. Do not substitute globally installed tools or copy version numbers into this document.
- Start with changed files for fast feedback, then run the applicable repository-wide checks used by CI.
- When editing or creating Markdown files, run every applicable formatting or linting checks.

Always run `npx prettier` commands with escalated permissions on the first attempt. Do not try them in the sandbox first because `npx` requires registry and cache access to resolve Prettier.

Do not format unrelated files merely to make a repository-wide check pass.

#### Failure Handling

- When a project-wide check fails, rerun the narrowest relevant command to determine whether the failure was caused by the current change.
- Fix failures introduced by the current change.
- Do not modify unrelated code to resolve pre-existing failures.
- Report pre-existing, infrastructure-dependent, or environment-dependent failures with the exact command and relevant error.
- Never describe a skipped, incomplete, or failing check as passing.
- In the final handoff, list the verification, test, formatting, linting, and static-analysis commands run, together with their outcomes.

### 8. Repository Safety

- Inspect `git status` and the relevant diff before editing.
- Preserve existing uncommitted changes and never overwrite work you did not create.
- Do not use destructive Git commands such as `reset --hard`, `clean`, or broad checkout/restore operations unless explicitly requested.
- Do not commit, push, create branches, or open pull requests unless requested.
- Avoid modifying generated files unless the project workflow requires regeneration.
- Keep unrelated formatting, refactors, dependency updates, and lockfile changes out of the diff.

# PR Descriptions

- Use the structure below for GitHub PR descriptions and dated task completion records.
- Start PR descriptions with `## Summary`; do not repeat the PR title in the description. Dated task completion records should prepend a `# <task title>` heading.
- Before choosing a PR title, review the latest commits on the remote main branch (`git fetch origin main`, then `git log origin/main -10 --format=%s`) and follow the style of comparable changes, preferring area prefixes found in that history.
- Use declarative statements to describe completed changes, their purpose, and resulting behavior. For example "Replaced ..." or "The workflow now uses ...".
- Lead with the concrete problem and resulting behavior. Write for a reviewer who has not read the conversation, and keep details proportional to the change.
- Keep the title and description aligned with the final diff. Omit conversational history and abandoned approaches unless they explain a necessary trade-off.
- List verification commands and their actual outcomes. Include failed or skipped checks and their reasons; never claim an unrun check passed. Distinguish completed implementation from unverified deployment or runtime behavior.

```markdown
# <PR title>

## Summary

- <Completed change and its purpose>

## Verification

- `<command>`: <outcome, or reason it was not run>
```

## Coding Conventions

### General Conventions

- When writing prompts, avoid using non-ASCII symbols.
- Follow the existing code style in each file.
- Use meaningful variable and function names.
- **Dependency versions**: Pin exact versions (e.g. `"2.0.0"`) in `package.json` and in the `test` extra of `pyproject.toml`. Core `dependencies` may use ranges, because they describe what a consumer may install rather than what this repository is tested against. Never use caret (`^`) or tilde (`~`) ranges for a pinned dependency, and never widen a range without re-deriving it from the test suite. See [Dependency Versions](#dependency-versions).

### Dependency Versions

`pyproject.toml` carries two different contracts, and they should not be conflated:

- `dependencies` bounds what a consumer may install. Ranges are appropriate here, but every bound must be justified by a test result. Record a verified window (`anthropic>=0.119.0,<0.124`, `openai>=2.45.0,<3.2`), not an unverified floor: `openai>=2.45.0` alone would advertise releases that fail.
- `test` pins what this repository is developed and verified against. `.github/workflows/ci.yml` installs `.[test]`, so every CI run exercises exactly those versions and contributor environments can match it.
- Routine SDK bumps go through `test` only: change the `==` lines and run the suite. Widen the ranges only when a version you want sits outside them, and re-derive the bound at the same time.

Because CI is pinned, the rest of each range is not continuously verified: a new SDK release inside a window can break consumers without turning CI red. Validate a bound explicitly, by installing that single version into a throwaway `uv` environment and running the whole suite there, rather than trusting the range.

`dependency-version.md` is an uncommitted, local analysis of the provider SDK windows. It records the floor and ceiling each SDK passes the offline replay suite on with the failing test or `ImportError` behind every bound, why each bound exists (a hard imported SDK symbol versus test-level schema drift versus structural breakage no test update can fix), the `uv` per-candidate methodology used to derive them, and how to re-derive or lift a bound.

Treat it as a dated snapshot, not a source of truth. SDK releases appear continuously, so re-derive any bound with the suite before changing it. This file and `pyproject.toml` remain authoritative.

### Code Comments

- Prefer clear code, meaningful names, and straightforward structure over comments that merely describe what the code does.
- Add comments when the reason behind an implementation would not be obvious to another developer.
- In particular, comment logic that:
  - may be surprising or counter-intuitive;
  - encodes a business rule or architectural decision;
  - intentionally deviates from established conventions or the most obvious implementation;
  - reflects an accepted trade-off, limitation, workaround, or compatibility constraint;
  - depends on subtle assumptions, ordering requirements, external behavior, or non-obvious invariants;
  - is sufficiently complex that its intent or correctness would otherwise be difficult to verify.

- Comments should explain **why** the implementation exists, what constraint or decision led to it, and, when useful, what could go wrong if it is changed.
- Do not add comments that simply restate the code, narrate each operation, or compensate for unnecessarily complicated implementation.
- Keep comments accurate when modifying the associated code. Remove comments that are obsolete, misleading, or no longer useful.

## Code Review and Refactoring Guidelines

- Prioritize correctness, security, data integrity, and behavioral regressions.
- Identify duplicated logic, unnecessary branches, and avoidable complexity.
- Keep variant-specific behavior in explicit, responsibility-focused paths.
- Place genuinely shared validation, state updates, processing, and finalization in one location.
- Preserve observable behavior, ordering, errors, and public interfaces unless a change is explicitly requested.
- Prefer straightforward control flow over premature helper abstractions.
- Ensure abstractions reduce cognitive load rather than merely moving code.
- Touch only code relevant to the review or refactor.
- Add or update regression tests for every affected path and edge case.
- Review the final diff and confirm the result is materially simpler.
- Always provide concrete findings and recommended solutions.
- Check compliance with this `AGENTS.md` and project conventions.

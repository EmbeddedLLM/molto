# Guide for AI agents

This `AGENTS.md` file provides comprehensive guidance for OpenAI Codex, Claude Code, and other AI agents working with this codebase.

The user's job as a senior software engineer is dependent on you implementing clean code with elegant architecture.

## Core Principles

- **No Laziness**: Find root causes. No temporary fixes. When fixing bugs or issues, prefer a clean solution over a hacky workaround. Senior developer standards.
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
- **Protect Secrets in Command Output**: Never print resolved configuration, environment variables, credentials, or command output that may expose secrets. When validating Compose files that use environment files.
- **The env config is only for running the tests**: It is never to be used for users requests.

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
- Do not create or modify task-tracking files for small or routine changes.
- Record only durable, project-specific lessons in `tasks/lessons.md`.
- Do not record one-off user preferences, speculative conclusions, or conversational details.
- A user correction is evidence to reconsider the approach, not an automatic reason to create a permanent rule.

### 4. Verification Before Done

- Run the strongest applicable verification available in the current environment.
- Never claim that a check passed unless it was actually run and passed.
- When a check cannot be run, report the exact command not run and the reason.
- Distinguish failures introduced by the current change from pre-existing or environment-dependent failures.
- Verify the behavior affected by the change, not merely that the code compiles.
- Add or update regression tests when fixing a reproducible defect. For any public provider behavior
  with an OpenAI SDK-equivalent contract, include cassette-backed parity
  coverage that compares Molto against parsed OpenAI SDK responses or events;
  IR and provider-setting unit tests may supplement this coverage, but must
  not replace it.
- Never run tests in cassette-recording mode. Cassettes must be supplied from
  an approved recorded source; test runs must replay them only.
- Compare behavior before and after the change when practical.
- Run the applicable checks defined in the Linting Workflow.
- Keep verification proportional to the scope and risk of the change. For behavior changes, run focused relevant tests; run `pytest` for broad changes or when the full offline replay suite is required.

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
- Use supplied logs, errors, and failing-test output to resolve the issue without rerunning tests
  locally.
- Zero context switching required from the user
- Fix failing CI tests autonomously from the available CI evidence; leave reruns to the user or CI

### 7. Linting Workflow

Run formatting, linting, and static analysis after implementation and before reporting completion.

The CI workflow is the source of truth. Before running checks, inspect `.github/workflows/lint.yml` and use its commands, installation method, configuration, and tool versions. Do not substitute globally installed tools or copy version numbers into this document.

Start with changed files for fast feedback, then run the applicable repository-wide checks used by CI.

#### Python

From the repository root:

1. Inspect the Ruff installation and check steps in `.github/workflows/lint.yml`.
2. Install or select Ruff using the workflow's current version constraint.
3. Confirm the selected version: `ruff --version`.
4. Format changed Python files: `ruff format <changed-python-files>`.
5. Check changed Python files: `ruff check <changed-python-files>`.
6. Run the same repository-wide Ruff format and lint checks configured in CI.

Do not run automatic fixes across the entire repository unless explicitly required by the task.

#### Repository formatting

From the repository root:

1. Inspect the Prettier check in `.github/workflows/lint.yml`.
2. Check changed files that are not excluded by `.prettierignore` with the workflow-selected Prettier version.
3. Run the same repository-wide formatting check configured in CI when appropriate.

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

## Coding Conventions

### General Conventions

- When writing prompts, avoid using non-ASCII symbols.
- Follow the existing code style in each file.
- Use meaningful variable and function names.
- **Pin dependency versions**: Use exact versions (e.g. `"2.0.0"`) in `package.json` and `pyproject.toml`. Never use caret (`^`) or tilde (`~`) ranges.

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

## Pull Request Review Guidelines

- Always prioritise any security issues
- Always provide solution or suggestions for any issues found
- Check whether the code aligns with Coding Conventions stated in `AGENTS.md`

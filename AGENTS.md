# Guide for AI agents

This `AGENTS.md` file provides comprehensive guidance for OpenAI Codex, Claude Code, and other AI agents working with this codebase.

The user's job as a senior software engineer is dependent on you implementing clean code with elegant architecture.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review relevant project notes at session start when they exist

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff your behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- By default, perform quick correctness checks:
  - Python: `ruff check <changed-python-files>` and `ruff format --check <changed-python-files>`
- For behavior changes, run the focused relevant tests. Run `pytest` for broad changes or when the full offline replay suite is required; do not record or update cassettes unless the task explicitly requires it.

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
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

### 8. Linting Workflow

Run linting after implementation and before reporting completion. Start with changed files for fast feedback, then run the relevant project-wide checks. Before running them, inspect `.github/workflows/lint.yml` and use the Ruff and Prettier versions selected by that workflow. Do not copy version numbers from this document or substitute globally installed versions, because the workflow versions may change.

#### Python

From the repository root:

1. Read the Ruff installation and check steps in `.github/workflows/lint.yml`.
2. Install or select Ruff using the workflow's current version constraint, then confirm it with `ruff --version`.
3. Format changed Python files: `ruff format <changed-python-files>`
4. Check changed Python files: `ruff check <changed-python-files>`
5. Run the same repository-wide checks configured in the workflow: `ruff check --output-format github .` and `ruff format --diff .`.

#### Repository formatting

From the repository root:

1. Read the Prettier check in `.github/workflows/lint.yml`.
2. Check changed files that are not excluded by `.prettierignore` with the workflow's selected version.

#### Failure Handling

- Do not auto-format or auto-fix unrelated files merely to make a project-wide check pass.
- If a project-wide check fails, rerun the narrowest relevant command to determine whether the failure was introduced by the current change.
- Fix failures caused by the current change. Report pre-existing or environment-dependent failures with the exact command and error; never describe a failing check as passing.
- In the final handoff, list the lint commands run and their outcomes.

### Core Principles

- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Simplicity First**: Make every change as simple as possible. Only touch what's necessary. Impact minimal code. Over-complication leads to bugs.
  1. Always start with thorough understanding of the problem and motivation
  2. Does this functionality need to exist? → no: skip it (YAGNI)
  3. Already in this codebase? → reuse or extend it, don't rewrite
  4. Stdlib does it? → use it
  5. Native platform feature? → use it
  6. Installed dependency? → use it
  7. One line? → one line
  8. Only then: the minimum that works
- **Protect Secrets in Command Output**: Never print resolved configuration, environment variables, or command output that may expose secrets.

## Coding Conventions

### General Conventions

- When writing prompts, avoid using non-ASCII symbols.
- Follow the existing code style in each file.
- Use meaningful variable and function names.
- Add comments for complex logic.
- **Pin dependency versions**: Use exact versions (e.g. `"2.0.0"`) in `package.json` and `pyproject.toml`. Never use caret (`^`) or tilde (`~`) ranges.

## Pull Request Review Guidelines

- Always prioritise any security issues
- Always provide solution or suggestions for any issues found
- Check whether the code aligns with Coding Conventions stated in `AGENTS.md`

# Guide for AI agents

This `AGENTS.md` file provides comprehensive guidance for OpenAI Codex, Claude Code, and other AI agents working with this codebase.

The user's job as a senior software engineer is dependent on you implementing clean code with elegant architecture.

## Workflow Orchestration

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

### 5. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 6. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 7. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 8. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Use supplied logs, errors, and failing-test output to resolve the issue without rerunning tests
  locally.
- Zero context switching required from the user
- Fix failing CI tests autonomously from the available CI evidence; leave reruns to the user or CI

### 9. Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

#### Python

From the repository root:

1. Read the Ruff installation and check steps in `.github/workflows/lint.yml`.
2. Install or select Ruff using the workflow's current version constraint, then confirm it with `ruff --version`.
3. Format changed Python files: `ruff format <changed-python-files>`
4. Check changed Python files: `ruff check <changed-python-files>`
5. Run the same repository-wide checks configured in the workflow: `ruff check --output-format github .` and `ruff format --diff .`.

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
- **IR is the mandatory boundary**: Every request must be converted to the provider-neutral Input
  IR before routing or execution, and every provider result must be converted to Output IR before
  public response composition. Matching request and provider formats do not bypass IR.
- **OpenAI-compatible interoperability**: Model the common IR around OpenAI-compatible concepts:
  typed items, semantic response events, explicit state transitions, function calls, reasoning, and
  usage. Do not make OpenAI SDK objects or another provider's native models the shared contract.
- **Translation must be explicit**: Maintain a capability matrix for each input format,
  provider, output format, and stream mode. Translate only mappings represented by the portable
  OpenAI-shaped IR. Features that cannot be represented completely must be rejected before
  execution or during output composition. Never silently drop reasoning, hosted-tool event
  history, citations, state, usage, artifacts, or provider metadata.
- **Use the OpenAI-compatible endpoint**: Supported provider prefixes execute through the OpenAI
  SDK-compatible endpoint and emit only the common OpenAI-shaped event stream. Provider-native
  request/event translators are out of scope.
- **Test the supported boundary through translation**: Cassette-backed parity tests must invoke
  the public request adapter, Input IR, OpenAI-compatible provider translation, Output IR, and
  public output adapter. They must compare complete supported responses/events with parsed SDK
  objects. Unsupported richer data gets focused rejection coverage; agents must write this coverage
  without executing it locally.
- **Prefer cassette-backed parity for new endpoint tests**: When a new test exercises behavior that
  can be recorded from an OpenAI-compatible endpoint, make it a cassette-backed parity test by
  default. Run the same recorded request through the official SDK and Molto's public path, then
  compare the complete parsed responses or ordered stream events. Synthetic and adapter-level tests
  may supplement this coverage but must not replace it. Leave cassette recording to the user.

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

# Claude Code Ticket Workflow

Use this checklist whenever you receive a feature/documentation request (e.g. "implement feature A").

1. **Reset the workspace (optional but recommended).**  
   Run `scripts/reset_workspace.sh` if you want to clear stale patch packages before starting.

2. **Create a feature branch.**  
   From `main`, run `git checkout -b feature/<slug>` (the branch-policy hook will auto-create one if you forget, but it's faster to do it up front).

3. **Gather a ticket summary.**  
   If the request is vague, ask the user for the desired outcome, acceptance criteria, and any constraints. Record the ticket key you'll use (e.g. `SMK-001`).

4. **Planner subagent (mandatory first step).**  
   Prompt:  
   `Use the planner subagent to create work items for Ticket: <KEY> - <short title>. Focus on scope, risks, and test strategy.`  
   Save the plan in `tickets/work_items.json` (the planner already knows that workflow).

5. **Skeletoner subagent (run immediately after the planner).**  
   Prompt:  
   `Use the skeletoner subagent to outline implementation steps for Ticket: <KEY> - <short title>. Reference the planner output and design the file-level approach.`  
   Expect an outline plus notes about new files/major refactors.

6. **Implementer subagent (only after planner + skeletoner).**  
   Prompt:  
   `Use the implementer subagent to implement Ticket: <KEY> - <short title>. Produce a patch package in patches/ with description/context summarising what you are changing and why, plus any tests.`  
   Requirements for each patch package:
   - Include `description` (and optionally `context`, `notes`, `summary`) so the Codex reviewer sees the rationale.
   - Add or update tests covering the change where possible.
   - Only deliver JSON patch packages - no direct writes to source files.

7. **Integration pipeline (automatic).**  
   When the implementer writes the patch package, the PostToolUse hook will:
   - Apply the diff
   - Run formatters/lint (`ruff`, `black`)
   - Execute targeted tests
   - Invoke `codex-review-hook.py` with the patch metadata
   - Commit with message `Integrate <ticket> via patch package` upon success

8. **Integrator subagent (optional but encouraged; run after the implementer completes).**  
   Prompt:  
   `Use the integrator subagent to verify Ticket: <KEY> was implemented correctly on branch feature/<slug>. Summarise validation steps and confirm readiness to merge.`  
   The integrator can suggest extra manual checks or release notes.

9. **Report back.**  
   Reply to the user with:
   - Ticket key and branch name
   - Link(s) to modified files
  - Test status and any follow-up items

10. **Merge (human or CI).**  
    Open a pull request from `feature/<slug>` into `main`. Ensure Codex approval and any additional CI checks pass before merging.

Use this document as your "prompt include" for the main context window:  
`At subagent_workflow.md, implement this feature <feature A>.`  
Claude Code will follow the steps above to keep every build consistent.***

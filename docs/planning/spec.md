# SMK-001: Smoketest - Patch Workflow Verification

## Overview
This is a minimal smoketest ticket designed to verify the patch-based workflow infrastructure. The task requires creating a single new file `smoketest.md` at the repository root with specific content.

## Goal
Validate that the subagent workflow (planner → skeletoner → implementer → integrator) can successfully:
1. Plan a trivial file creation task
2. Generate a patch package
3. Apply the patch through the integration pipeline
4. Verify the result

## Scope

### In Scope
- Create `/home/donaldshen27/projects/donald_trading_model/smoketest.md`
- Write exactly one line: "new workflow smoketest"
- Validate file creation through git status
- Confirm no side effects on existing codebase

### Out of Scope
- Any modifications to existing files
- Testing complex logic or multi-file changes
- Integration with CI/CD pipelines
- Documentation updates beyond the smoketest file itself

## Non-Goals
- Performance optimization
- Error handling or edge cases
- Multi-region support
- Backward compatibility (this is a net-new file)

## Modules & Interfaces

### Module: File Creation
**Purpose:** Create a single markdown file at repository root

**Input:** None (hardcoded content)

**Output:** File at `/home/donaldshen27/projects/donald_trading_model/smoketest.md`

**Contract:**
```yaml
file_path: /home/donaldshen27/projects/donald_trading_model/smoketest.md
content: "new workflow smoketest\n"
encoding: utf-8
permissions: 0644 (default)
```

**Dependencies:** None

**Risk Assessment:**
- **LOW:** File creation is atomic operation
- **LOW:** No existing file will be overwritten (new file)
- **LOW:** No code dependencies or imports required

## Architecture Decisions

### AD-1: Single File Approach
**Decision:** Use direct file write rather than template system

**Rationale:** 
- Simplicity: No templating overhead for static content
- Testability: Easy to verify exact content match
- Clarity: Demonstrates minimal viable patch workflow

**Alternatives Considered:**
- Template-based approach: Rejected due to unnecessary complexity
- Multi-file creation: Rejected to keep smoketest focused

### AD-2: Root-Level Placement
**Decision:** Place file at repository root rather than docs/ or tests/

**Rationale:**
- Visibility: Easily discoverable by developers
- Isolation: No impact on organized directory structures
- Cleanup: Can be git-ignored or removed after workflow validation

## Test Strategy

### Verification Criteria
1. **File Exists:** `test -f /home/donaldshen27/projects/donald_trading_model/smoketest.md`
2. **Content Match:** `[ "$(cat smoketest.md)" = "new workflow smoketest" ]`
3. **Git Status:** File appears as untracked or staged
4. **No Side Effects:** `git status` shows only smoketest.md changed

### Test Execution
```bash
# Verify file creation
ls -la /home/donaldshen27/projects/donald_trading_model/smoketest.md

# Verify content
cat /home/donaldshen27/projects/donald_trading_model/smoketest.md

# Verify git tracking
git -C /home/donaldshen27/projects/donald_trading_model status --short
```

### Success Criteria
- All 4 verification criteria pass
- No errors during patch application
- Integrator subagent confirms readiness

### Failure Scenarios
- **File not created:** Check patch package syntax
- **Wrong content:** Verify patch diff correctness
- **Permission errors:** Check filesystem permissions
- **Unexpected changes:** Review patch application logic

## Risks & Mitigations

### Risk 1: Patch Application Failure
**Likelihood:** Low  
**Impact:** Medium (blocks workflow validation)  
**Mitigation:** Use simple file creation (not modification) to minimize failure modes

### Risk 2: Git Hook Interference
**Likelihood:** Medium  
**Impact:** Low (may block commit but won't corrupt files)  
**Mitigation:** Ensure branch-policy hook accepts feature branches; verify via `git status`

### Risk 3: File Already Exists
**Likelihood:** Very Low  
**Impact:** Low (implementer will see error)  
**Mitigation:** Check file existence before patch generation; document expected state

## Dependencies

### External
- Git repository initialized (already met)
- Write permissions on repository root (assumed)

### Internal
- Patch integration pipeline configured (per subagent_workflow.md)
- Implementer subagent available

### Blocked By
None (this is a standalone task)

### Blocks
None (this is a validation task, not a feature dependency)

## Acceptance Criteria

1. File `/home/donaldshen27/projects/donald_trading_model/smoketest.md` exists
2. File contains exactly: `new workflow smoketest\n`
3. `git status` shows the file as tracked/staged
4. No other files modified (verified via `git diff --name-only`)
5. Implementer produces valid patch package in `patches/`
6. Integrator confirms successful validation

## Implementation Notes

### For Skeletoner
- Single work item: CREATE_SMOKETEST_FILE
- No refactoring or multi-step coordination required
- Document patch package location convention

### For Implementer
- Use JSON patch package format with `description` field
- Include test commands in patch metadata
- Ensure newline at end of file for POSIX compliance

### For Integrator
- Run all 4 verification criteria
- Confirm git branch is feature/smk-001 (or similar)
- Verify no unexpected files in `git status`

## Timeline Estimate
- Planner: Complete (this document)
- Skeletoner: ~1 minute (single work item)
- Implementer: ~2 minutes (patch generation)
- Integration: ~30 seconds (automated)
- Integrator: ~1 minute (verification)

**Total:** ~5 minutes end-to-end

## References
- Workflow: `/home/donaldshen27/projects/donald_trading_model/subagent_workflow.md`
- Repository root: `/home/donaldshen27/projects/donald_trading_model/`
- Patch directory: `/home/donaldshen27/projects/donald_trading_model/patches/`

---
**Spec Version:** 1.0  
**Created:** 2025-10-13  
**Ticket:** SMK-001

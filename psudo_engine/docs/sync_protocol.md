# Sync Protocol Documentation

## Overview

This document describes the synchronization protocol between the two repositories:
- **psudo-engine** (Training & Model Repository)
- **psudo-harness** (Execution & Transpiler Repository)

Both repositories share a single source of truth: `psudo-spec.json` (the contract).

---

## Repository Structure

```
psudo-engine/                 # Repository A: Training
├── psudo_engine/
│   ├── data/
│   │   ├── psudo-spec.json   # Vendored copy of shared contract
│   │   ├── base_grammar.jsonl
│   │   └── feedback_loop/
│   │       ├── corrections_001.jsonl
│   │       └── ...
│   ├── training/
│   │   └── schema_check.py   # Version validation
│   └── ...

psudo-harness/                # Repository B: Execution/Transpiler
├── psudo_harness/
│   ├── data/
│   │   └── psudo-spec.json   # Same shared contract (submodule or copy)
│   ├── transpiler/
│   │   └── schema_check.rs   # Version validation (Rust)
│   └── ...
```

---

## Shared Contract: psudo-spec.json

**Location**: `psudo_engine/data/psudo-spec.json` (source of truth)

**Contents**:
- Grammar specification
- Keywords and builtins
- AST node definitions
- Type system
- Transpiler configuration
- Feedback format
- Dataset schema

**Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- Current: `1.0.0`

---

## Sync Mechanisms

### 1. Git Submodule (Recommended)

Add `psudo-spec.json` as a git submodule in both repositories:

```bash
# In psudo-engine
git submodule add https://github.com/user/psudo-spec.git psudo_engine/data/psudo-spec

# In psudo-harness
git submodule add https://github.com/user/psudo-spec.git psudo_harness/data/psudo-spec
```

**Workflow**:
1. Make changes to `psudo-spec` repo
2. Commit and push to `psudo-spec`
3. Update submodules in both repos:
   ```bash
   git submodule update --remote
   git commit -m "Update psudo-spec to v1.1.0"
   ```

### 2. Vendored Copy with CI Check

If submodules aren't preferred, vendor the file and use CI to verify sync:

**`.github/workflows/sync-check.yml`**:
```yaml
name: Sync Check
on: [push, pull_request]
jobs:
  check-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check psudo-spec sync
        run: |
          # Compare with upstream spec repo
          curl -s https://raw.githubusercontent.com/user/psudo-spec/main/psudo-spec.json > upstream.json
          diff -u upstream.json psudo_engine/data/psudo-spec.json || {
            echo "psudo-spec out of sync!"
            exit 1
          }
```

### 3. Manual Sync with Validation

For simpler workflows, manual sync with validation:

```bash
# Sync script (sync-spec.sh)
#!/bin/bash
set -e

SPEC_URL="https://raw.githubusercontent.com/user/psudo-spec/main/psudo-spec.json"
TARGET="psudo_engine/data/psudo-spec.json"

echo "Fetching latest psudo-spec..."
curl -s "$SPEC_URL" > "$TARGET.new"

# Validate JSON
python -m json.tool "$TARGET.new" > /dev/null

# Check version
VERSION=$(python -c "import json; print(json.load(open('$TARGET.new'))['version'])")
echo "New version: $VERSION"

# Backup and replace
cp "$TARGET" "$TARGET.bak"
mv "$TARGET.new" "$TARGET"

# Run validation
python -m psudo_engine.training.schema_check "$TARGET"

echo "Sync complete: $VERSION"
```

---

## Version Validation

Both repositories must validate schema version on startup.

### Training Pipeline (Python)

```python
# In train.py
from psudo_engine.training.schema_check import check_schema_version, SchemaValidator

config = OmegaConf.load("config/training.yaml")
if not check_schema_version(config, "training"):
    sys.exit(1)

# Or with full report
validator = SchemaValidator()
validator.print_compatibility_report(
    training_config=config,
    transpiler_config=harness_config
)
```

### Transpiler (Rust)

```rust
// In transpiler/src/schema_check.rs
use std::fs;

fn validate_schema_version(config: &Config) -> Result<(), SchemaError> {
    let spec = fs::read_to_string("data/psudo-spec.json")?;
    let spec: Spec = serde_json::from_str(&spec)?;
    
    let expected = config.transpiler.spec_version;
    if spec.version != expected {
        return Err(SchemaError::VersionMismatch {
            expected,
            actual: spec.version,
        });
    }
    Ok(())
}
```

---

## Feedback Loop Protocol

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model     │────▶│  Transpiler │────▶│  Compiler   │
│  (psudo)    │     │  (Rust)     │     │  (cargo)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  LSP/Rustc  │     │   Errors    │
                    │  Diagnostics│     │  (JSON)     │
                    └─────────────┘     └─────────────┘
                           │                   │
                           └─────────┬─────────┘
                                     ▼
                          ┌─────────────────┐
                          │  Correction     │
                          │  Pair Builder   │
                          └─────────────────┘
                                     │
                                     ▼
                          ┌─────────────────┐
                          │  Feedback JSONL │
                          │  (per spec)     │
                          └─────────────────┘
                                     │
                                     ▼
                          ┌─────────────────┐
                          │  psudo-engine/  │
                          │  data/feedback  │
                          └─────────────────┘
```

### Correction Pair Format

Per `psudo-spec.json` feedback format:

```json
{
  "prompt": "Write a function to read a file",
  "faulty_code": "fn read(path: string) -> string { open(path) }",
  "error_message": "error[E0433]: failed to resolve: use of undeclared crate or module `open`\n  --> src/main.rs:2:30\n   |\n2  |     open(path)\n   |     ^^^^ use of undeclared crate or module `open`\n   |\n   = help: consider importing `std::fs::read_to_string`",
  "corrected_code": "fn read_file(path: string) -> string { read_file(path) }",
  "error_type": "E0433",
  "line_number": 2,
  "column_number": 30
}
```

### Feedback Ingestion

**Harness writes to local buffer**:
```
psudo-harness/data/feedback_loop/
├── corrections_001.jsonl
├── corrections_002.jsonl
└── ...
```

**Sync to training repo**:
```bash
# Periodic sync (cron or CI)
rsync -av psudo-harness/data/feedback_loop/ psudo-engine/data/feedback_loop/

# Or push to shared storage (S3, GCS, etc.)
aws s3 sync psudo-harness/data/feedback_loop/ s3://psudo-feedback/
aws s3 sync s3://psudo-feedback/ psudo-engine/data/feedback_loop/
```

**Training reads feedback**:
```python
# In ternary_trainer.py
from psudo_engine.training.psudo_dataset import FeedbackDataset

feedback_dataset = FeedbackDataset(
    base_dataset=train_dataset,
    feedback_path="data/feedback_loop/",
    tokenizer=tokenizer,
    feedback_ratio=0.3
)
```

---

## Release Procedure

### 1. Schema Changes

When adding new syntax/features:

1. Update `psudo-spec.json` (bump version)
2. Update transpiler to handle new syntax
3. Update training data generator
4. Run full test suite
5. Release new version

### 2. Version Bumping

```bash
# Patch: bug fixes, no new syntax
# Minor: new builtins, keywords (backward compatible)
# Major: breaking grammar changes

# Example: Minor version bump
# Edit psudo-spec.json: "version": "1.1.0"
git commit -am "feat: add async/await support"
git tag v1.1.0
git push origin v1.1.0
```

### 3. Coordinated Release

Both repos must release together:

```bash
# In psudo-engine
git submodule update --remote
git commit -am "chore: update psudo-spec to v1.1.0"
git tag v1.1.0-engine
git push origin v1.1.0-engine

# In psudo-harness
git submodule update --remote
git commit -am "chore: update psudo-spec to v1.1.0"
git tag v1.1.0-harness
git push origin v1.1.0-harness
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/sync.yml
name: Sync & Validate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate-schema:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      
      - name: Validate Python schema
        run: |
          cd psudo-engine
          python -m psudo_engine.training.schema_check psudo_engine/data/psudo-spec.json
      
      - name: Validate Rust schema
        run: |
          cd psudo-harness
          cargo test schema_check
  
  test-feedback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      
      - name: Generate test feedback
        run: |
          # Run model, transpile, capture errors
          python -m psudo_engine.inference.cli generate ...
      
      - name: Validate feedback format
        run: |
          python -m psudo_engine.training.schema_check --feedback data/feedback_loop/
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| Version mismatch | Submodule not updated | `git submodule update --remote` |
| Feedback not ingested | Wrong path in config | Check `feedback_path` in training.yaml |
| Transpiler errors | Schema not synced | Update submodule, rebuild transpiler |
| Validation fails | Spec file corrupted | Re-fetch from source |

### Debug Commands

```bash
# Check schema version
python -c "import json; print(json.load(open('psudo_engine/data/psudo-spec.json'))['version'])"

# Validate schema
python -m psudo_engine.training.schema_check psudo_engine/data/psudo-spec.json

# Check feedback format
python -c "
import json
for line in open('psudo_engine/data/feedback_loop/corrections_001.jsonl'):
    sample = json.loads(line)
    assert 'faulty_code' in sample and 'corrected_code' in sample
print('All feedback samples valid')
"
```

---

## Security Considerations

1. **Feedback Data**: May contain user code - sanitize before logging
2. **Schema Source**: Verify HTTPS/TLS when fetching remote spec
3. **Submodule Trust**: Only use trusted upstream spec repository
4. **CI Secrets**: Store any API tokens in GitHub Secrets, not in repo

---

## Migration Guide

### From Manual to Submodule

```bash
# 1. Backup current spec
cp psudo_engine/data/psudo-spec.json psudo_engine/data/psudo-spec.json.bak

# 2. Remove vendored file
git rm psudo_engine/data/psudo-spec.json

# 3. Add submodule
git submodule add https://github.com/user/psudo-spec.git psudo_engine/data/psudo-spec

# 4. Commit
git commit -m "chore: migrate psudo-spec to submodule"
```

---

## Contacts

- **Schema Maintainer**: [Team/Individual]
- **Training Pipeline**: [Team/Individual]
- **Transpiler**: [Team/Individual]

---

*Last Updated: 2026-10-27*
*Version: 1.0.0*
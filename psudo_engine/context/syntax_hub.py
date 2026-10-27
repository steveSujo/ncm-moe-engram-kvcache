"""
Syntax Hub for psudo-Engine Context Subsystem.

Manages modular Markdown syntax rule files for Python, Rust, and psudo builtins.
Provides indexing and retrieval of syntax rules for context injection.
"""

import contextlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SyntaxRule:
    """A single syntax rule from a Markdown file."""
    id: str
    category: str  # 'builtin', 'control_flow', 'type', 'pattern', 'idiom'
    language: str  # 'python', 'rust', 'psudo', 'generic'
    title: str
    description: str
    pattern: str  # Code pattern or example
    tags: list[str] = field(default_factory=list)
    priority: int = 1  # Higher = more important for context
    file_source: str = ""
    line_start: int = 0
    line_end: int = 0


@dataclass
class SyntaxFile:
    """A Markdown syntax rule file."""
    path: str
    language: str
    category: str
    rules: list[SyntaxRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SyntaxHub:
    """
    Syntax Hub - Manages modular Markdown syntax rule files.

    Loads, indexes, and retrieves syntax rules for context injection.
    Rules are organized by language and category with priority scoring.
    """

    # Default rule directories
    DEFAULT_DIRS = {
        'python': 'syntax_rules/python',
        'rust': 'syntax_rules/rust',
        'psudo': 'syntax_rules/psudo',
        'generic': 'syntax_rules/generic',
    }

    # Rule categories
    CATEGORIES = [
        'builtin',      # Language builtins (print, len, etc.)
        'control_flow', # if, while, for, match, etc.
        'type',         # Type definitions, annotations
        'pattern',      # Common patterns (iterators, error handling)
        'idiom',        # Language-specific idioms
        'stdlib',       # Standard library usage
        'psudo',        # psudo-specific syntax
    ]

    def __init__(self, rules_dir: str | None = None):
        if rules_dir:
            self.rules_dir = Path(rules_dir)
        else:
            # Default to psudo_engine/context/syntax_rules relative to this file
            self.rules_dir = Path(__file__).parent / 'syntax_rules'

        self.rules_dir.mkdir(parents=True, exist_ok=True)

        self.files: dict[str, SyntaxFile] = {}
        self.rules_by_id: dict[str, SyntaxRule] = {}
        self.rules_by_category: dict[str, list[SyntaxRule]] = defaultdict(list)
        self.rules_by_language: dict[str, list[SyntaxRule]] = defaultdict(list)
        self.rules_by_tag: dict[str, list[SyntaxRule]] = defaultdict(list)

        self._initialized = False

    def initialize(self, create_defaults: bool = True):
        """Load all syntax rule files from disk."""
        if self._initialized:
            return

        if create_defaults:
            self._create_default_rules()

        self._load_all_rules()
        self._initialized = True

    def _create_default_rules(self):
        """Create default syntax rule files if they don't exist."""
        # Python builtins
        self._write_default_file('python/builtins.md', """
---
language: python
category: builtin
---

# Python Built-in Functions

## print
**Pattern:** `print(*objects, sep=' ', end='\\n', file=sys.stdout, flush=False)`
**Description:** Prints objects to the text stream file, separated by sep and followed by end.
**Tags:** io, output, debugging
**Priority:** 10

## len
**Pattern:** `len(obj) -> int`
**Description:** Returns the number of items in an object.
**Tags:** collections, sequence, mapping
**Priority:** 10

## range
**Pattern:** `range(stop) -> range` / `range(start, stop[, step]) -> range`
**Description:** Returns a sequence of numbers from start (default 0) to stop, incrementing by step (default 1).
**Tags:** iteration, loops, sequences
**Priority:** 9

## enumerate
**Pattern:** `enumerate(iterable, start=0) -> enumerate`
**Description:** Returns an iterator yielding (index, value) pairs from iterable.
**Tags:** iteration, loops, indexing
**Priority:** 8

## zip
**Pattern:** `zip(*iterables, strict=False) -> zip`
**Description:** Returns an iterator of tuples, where the i-th tuple contains the i-th element from each iterable.
**Tags:** iteration, parallel, tuples
**Priority:** 8

## map / filter
**Pattern:** `map(function, iterable, ...) -> map` / `filter(function, iterable) -> filter`
**Description:** Apply function to every item of iterable / filter items where function returns True.
**Tags:** functional, transformation, filtering
**Priority:** 7

## open
**Pattern:** `open(file, mode='r', encoding=None) -> file object`
**Description:** Opens file and returns a file object. Use with `with` statement.
**Tags:** io, file, context_manager
**Priority:** 9
""")

        # Python control flow
        self._write_default_file('python/control_flow.md', """
---
language: python
category: control_flow
---

# Python Control Flow

## if / elif / else
**Pattern:**
```python
if condition:
    # block
elif other_condition:
    # block
else:
    # block
```
**Description:** Conditional execution. Conditions evaluate to boolean.
**Tags:** conditional, branching
**Priority:** 10

## while loop
**Pattern:**
```python
while condition:
    # block
else:
    # executed when condition becomes False (not on break)
```
**Description:** Repeatedly executes block while condition is True.
**Tags:** loop, iteration
**Priority:** 10

## for loop
**Pattern:**
```python
for item in iterable:
    # block
else:
    # executed when loop completes (not on break)
```
**Description:** Iterates over elements of iterable.
**Tags:** loop, iteration, sequence
**Priority:** 10

## break / continue
**Pattern:**
```python
for item in items:
    if condition:
        break  # exit loop
    if other_condition:
        continue  # skip to next iteration
```
**Description:** break exits the loop; continue skips to next iteration.
**Tags:** loop, control
**Priority:** 9

## try / except / else / finally
**Pattern:**
```python
try:
    # risky code
except SpecificError as e:
    # handle error
except Exception:
    # handle other errors
else:
    # no exception raised
finally:
    # always executed
```
**Description:** Exception handling. else runs if no exception; finally always runs.
**Tags:** error_handling, exceptions
**Priority:** 9

## match / case (Python 3.10+)
**Pattern:**
```python
match value:
    case pattern1:
        # block
    case pattern2 if guard:
        # block
    case _:
        # default
```
**Description:** Structural pattern matching.
**Tags:** pattern_matching, branching
**Priority': 8
""")

        # Rust builtins
        self._write_default_file('rust/builtins.md', """
---
language: rust
category: builtin
---

# Rust Built-in / Standard Library

## println! / eprintln!
**Pattern:** `println!("{}", value)` / `eprintln!("{}", value)`
**Description:** Prints to stdout/stderr with formatting. Uses format! syntax.
**Tags:** io, output, debugging, macro
**Priority:** 10

## format!
**Pattern:** `format!("{}", value) -> String`
**Description:** Creates a String using formatting. Similar to println! but returns String.
**Tags:** formatting, string, macro
**Priority:** 9

## vec! / String::new
**Pattern:** `vec![1, 2, 3]` / `String::new()`
**Description:** Create vector / empty string.
**Tags:** collections, initialization, macro
**Priority:** 9

## Option / Result
**Pattern:** `Option<T>` / `Result<T, E>`
**Description:** Option for nullable values; Result for error handling.
**Tags:** error_handling, option, result, type
**Priority:** 10

## match
**Pattern:**
```rust
match value {
    Some(x) => { /* ... */ }
    None => { /* ... */ }
    _ => { /* catch-all */ }
}
```
**Description:** Exhaustive pattern matching. Must handle all cases.
**Tags:** pattern_matching, control_flow
**Priority:** 10

## if let / while let
**Pattern:**
```rust
if let Some(x) = optional {
    // use x
}
```
**Description:** Pattern matching in conditionals.
**Tags:** pattern_matching, conditional
**Priority:** 8

## ? operator
**Pattern:** `let value = fallible_function()?;`
**Description:** Propagates errors. Returns Err from function if Err, unwraps if Ok.
**Tags:** error_handling, propagation
**Priority:** 10

## iterators: map, filter, collect
**Pattern:**
```rust
let result: Vec<i32> = vec.iter()
    .map(|x| x * 2)
    .filter(|x| *x > 10)
    .collect();
```
**Description:** Iterator chain for transformation and filtering.
**Tags:** iterator, functional, collections
**Priority:** 9
""")

        # Rust control flow
        self._write_default_file('rust/control_flow.md', """
---
language: rust
category: control_flow
---

# Rust Control Flow

## if / else
**Pattern:**
```rust
if condition {
    // block
} else if other {
    // block
} else {
    // block
}
```
**Description:** Conditional. Condition must be bool. Blocks are expressions.
**Tags:** conditional, branching, expression
**Priority:** 10

## loop / while / for
**Pattern:**
```rust
loop {
    // infinite loop
    break;  // exit
}

while condition {
    // block
}

for item in iterable {
    // block
}
```
**Description:** loop is infinite; while checks condition; for iterates.
**Tags:** loop, iteration
**Priority:** 10

## break / continue with labels
**Pattern:**
```rust
'outer: for i in 0..10 {
    for j in 0..10 {
        if condition {
            break 'outer;  // breaks outer loop
        }
    }
}
```
**Description:** Labels allow breaking/continuing outer loops.
**Tags:** loop, control, labels
**Priority:** 8

## return
**Pattern:** `return value;` or implicit (last expression)
**Description:** Explicit return or implicit return from block expression.
**Tags:** function, control_flow
**Priority:** 9
""")

        # psudo syntax
        self._write_default_file('psudo/syntax.md', """
---
language: psudo
category: psudo
---

# psudo Language Syntax

## Variable Assignment
**Pattern:** `let name = value` / `name = value`
**Description:** Binds a value to a name. Type inferred or explicit with `let name: type = value`.
**Tags:** variable, assignment, binding
**Priority:** 10

## Function Definition
**Pattern:**
```
fn name(param: type, ...) -> return_type {
    // body
    return value
}
```
**Description:** Defines a function with typed parameters and return type.
**Tags:** function, definition, types
**Priority:** 10

## Control Flow - if
**Pattern:**
```
if condition {
    // then branch
} else {
    // else branch
}
```
**Description:** Conditional execution. Condition must be bool.
**Tags:** conditional, branching
**Priority:** 10

## Control Flow - while
**Pattern:**
```
while condition {
    // body
}
```
**Description:** Repeats while condition is true.
**Tags:** loop, iteration
**Priority:** 10

## Control Flow - for
**Pattern:**
```
for item in iterable {
    // body
}
```
**Description:** Iterates over iterable (range, list, etc.).
**Tags:** loop, iteration
**Priority:** 10

## Built-in: read_file
**Pattern:**
```psudo
read_file(path: string) -> string
```
**Description:** Reads entire file content as string. Use `read_file(path)` to read a file. Aliases: read file, read_file.
**Tags:** builtin, io, file
**Priority:** 10

## Built-in: write_file
**Pattern:** `write_file(path: string, content: string) -> ()`
**Description:** Writes content to file, creating directories as needed.
**Tags:** builtin, io, file
**Priority:** 10

## Built-in: print
**Pattern:** `print(value: any) -> ()`
**Description:** Prints value to stdout with default formatting.
**Tags:** builtin, io, output
**Priority:** 9

## Type Annotations
**Pattern:** `let name: type = value` / `fn name(param: type) -> return_type`
**Description:** Explicit type annotations. Primitive types: int, float, string, bool. Containers: list<T>, dict<K, V>.
**Tags:** types, annotations
**Priority:** 8

## Comments
**Pattern:** `# comment` / `## doc comment`
**Description:** Single-line comments. Double-hash for documentation comments.
**Tags:** comments, documentation
**Priority:** 5
""")

        # Generic patterns
        self._write_default_file('generic/patterns.md', """
---
language: generic
category: pattern
---

# Generic Programming Patterns

## Error Handling with Result
**Pattern:**
```
result = operation()
if result.is_ok():
    value = result.unwrap()
else:
    handle_error(result.err())
```
**Description:** Explicit error checking instead of exceptions.
**Tags:** error_handling, result, pattern
**Priority:** 9

## Builder Pattern
**Pattern:**
```
builder = Builder::new()
    .with_option(value)
    .with_other(value)
result = builder.build()
```
**Description:** Fluent builder for complex object construction.
**Tags:** builder, construction, pattern
**Priority:** 7

## Iterator Chain
**Pattern:**
```
result = collection
    .filter(predicate)
    .map(transform)
    .collect()
```
**Description:** Chain iterator operations for data transformation.
**Tags:** iterator, functional, chain
**Priority:** 8

## Resource Acquisition Is Initialization (RAII)
**Pattern:**
```
resource = acquire()
try:
    use(resource)
finally:
    release(resource)
```
**Description:** Ensure resource cleanup with try/finally or context managers.
**Tags:** resource, cleanup, pattern
**Priority:** 8
""")

    def _write_default_file(self, rel_path: str, content: str):
        """Write a default syntax rule file if it doesn't exist."""
        file_path = self.rules_dir / rel_path
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content.strip() + '\n', encoding='utf-8')

    def _load_all_rules(self):
        """Load all .md files from rules directory."""
        self.files.clear()
        self.rules_by_id.clear()
        self.rules_by_category.clear()
        self.rules_by_language.clear()
        self.rules_by_tag.clear()

        for md_file in self.rules_dir.rglob('*.md'):
            self._load_file(md_file)

    def _load_file(self, file_path: Path):
        """Load and parse a single Markdown rule file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return

        # Parse frontmatter
        rel_path = file_path.relative_to(self.rules_dir)
        metadata = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    metadata = json.loads(parts[1]) if parts[1].strip() else {}
                    # Also support YAML-like frontmatter
                    if not metadata:
                        metadata = self._parse_simple_frontmatter(parts[1])
                except (json.JSONDecodeError, ValueError):
                    metadata = self._parse_simple_frontmatter(parts[1])
                body = parts[2]

        language = metadata.get('language', 'generic')
        category = metadata.get('category', 'pattern')

        # Parse rules from body
        rules = self._parse_rules_from_markdown(body, str(rel_path), language, category)

        syntax_file = SyntaxFile(
            path=str(rel_path),
            language=language,
            category=category,
            rules=rules,
            metadata=metadata
        )

        self.files[str(rel_path)] = syntax_file

        for rule in rules:
            self.rules_by_id[rule.id] = rule
            self.rules_by_category[rule.category].append(rule)
            self.rules_by_language[rule.language].append(rule)
            for tag in rule.tags:
                self.rules_by_tag[tag].append(rule)

    def _parse_simple_frontmatter(self, text: str) -> dict[str, Any]:
        """Parse simple key: value frontmatter."""
        metadata = {}
        for line in text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _parse_rules_from_markdown(
        self,
        content: str,
        file_source: str,
        default_language: str,
        default_category: str
    ) -> list[SyntaxRule]:
        """Parse syntax rules from Markdown content."""
        rules = []
        lines = content.split('\n')

        current_rule: dict[str, Any] = {}
        in_code_block = False
        code_lines = []
        rule_counter = 0

        for _i, line in enumerate(lines):
            stripped = line.strip()

            # Detect rule header (## Title)
            if stripped.startswith('## ') and not in_code_block:
                # Save previous rule
                if current_rule:
                    rule = self._finalize_rule(current_rule, code_lines, file_source, rule_counter)
                    if rule:
                        rules.append(rule)
                    rule_counter += 1

                # Start new rule
                current_rule = {
                    'title': stripped[3:].strip(),
                    'description': '',
                    'pattern': '',
                    'tags': [],
                    'priority': 1,
                    'category': default_category,
                    'language': default_language,
                }
                code_lines = []

            # Detect code blocks
            elif stripped.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block and code_lines:
                    current_rule['pattern'] = '\n'.join(code_lines)
                    code_lines = []
            elif in_code_block:
                code_lines.append(line)

            # Detect tags
            elif stripped.startswith('**Tags:**'):
                tags_str = stripped[9:].strip()
                current_rule['tags'] = [t.strip() for t in tags_str.split(',')]

            # Detect priority
            elif stripped.startswith('**Priority:**'):
                with contextlib.suppress(BaseException):
                    current_rule['priority'] = int(stripped[12:].strip())

            # Detect description
            elif stripped.startswith('**Description:**'):
                current_rule['description'] = stripped[14:].strip()
            elif current_rule and stripped and not stripped.startswith('**') and not current_rule.get('description'):
                current_rule['description'] = stripped

        # Save last rule
        if current_rule:
            rule = self._finalize_rule(current_rule, code_lines, file_source, rule_counter)
            if rule:
                rules.append(rule)

        return rules

    def _finalize_rule(
        self,
        rule_data: dict[str, Any],
        code_lines: list[str],
        file_source: str,
        counter: int
    ) -> SyntaxRule | None:
        """Create SyntaxRule from parsed data."""
        if not rule_data.get('title'):
            return None

        # Use code block as pattern, or generate from title
        pattern = '\n'.join(code_lines) if code_lines else rule_data.get('pattern', '')

        # Generate ID
        rule_id = f"{rule_data['language']}_{rule_data['category']}_{rule_data['title'].lower().replace(' ', '_')}"
        rule_id = re.sub(r'[^a-z0-9_]', '', rule_id)

        return SyntaxRule(
            id=rule_id,
            category=rule_data.get('category', 'pattern'),
            language=rule_data.get('language', 'generic'),
            title=rule_data['title'],
            description=rule_data.get('description', ''),
            pattern=pattern,
            tags=rule_data.get('tags', []),
            priority=rule_data.get('priority', 1),
            file_source=file_source,
            line_start=counter * 100,  # Approximate
            line_end=counter * 100 + 50
        )

    def get_rules(
        self,
        language: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        max_rules: int = 50
    ) -> list[SyntaxRule]:
        """
        Get syntax rules filtered by criteria.

        Args:
            language: Filter by language ('python', 'rust', 'psudo', 'generic')
            category: Filter by category
            tags: Filter by tags (any match)
            max_rules: Maximum number of rules to return

        Returns:
            List of matching SyntaxRule, sorted by priority (descending)
        """
        candidates = list(self.rules_by_id.values())

        if language:
            candidates = [r for r in candidates if r.language == language]

        if category:
            candidates = [r for r in candidates if r.category == category]

        if tags:
            tag_set = set(tags)
            candidates = [r for r in candidates if tag_set.intersection(r.tags)]

        # Sort by priority (descending)
        candidates.sort(key=lambda r: -r.priority)

        return candidates[:max_rules]

    def search_rules(self, query: str, max_rules: int = 20) -> list[SyntaxRule]:
        """Search rules by title, description, or pattern."""
        query_lower = query.lower()
        matches = []

        for rule in self.rules_by_id.values():
            score = 0
            if query_lower in rule.title.lower():
                score += 10
            if query_lower in rule.description.lower():
                score += 5
            if query_lower in rule.pattern.lower():
                score += 3
            if any(query_lower in tag.lower() for tag in rule.tags):
                score += 5

            if score > 0:
                matches.append((score, rule))

        matches.sort(key=lambda x: -x[0])
        return [r for _, r in matches[:max_rules]]

    def get_context_for_task(
        self,
        task_description: str,
        language: str = 'psudo',
        max_tokens: int = 1500
    ) -> str:
        """
        Get formatted context string for a task.

        Args:
            task_description: Description of the task
            language: Target language for context
            max_tokens: Approximate token budget

        Returns:
            Formatted context string with relevant syntax rules
        """
        # Search for relevant rules
        rules = self.search_rules(task_description, max_rules=30)

        # Also get language-specific high-priority rules
        lang_rules = self.get_rules(language=language, max_rules=15)

        # Combine and deduplicate
        seen = set()
        combined = []
        for r in rules + lang_rules:
            if r.id not in seen:
                seen.add(r.id)
                combined.append(r)

        # Sort by priority
        combined.sort(key=lambda r: -r.priority)

        # Build context string within token budget
        context_parts = []
        estimated_tokens = 0

        for rule in combined:
            rule_text = self._format_rule(rule)
            rule_tokens = len(rule_text) // 4  # Rough estimate

            if estimated_tokens + rule_tokens > max_tokens:
                break

            context_parts.append(rule_text)
            estimated_tokens += rule_tokens

        return '\n\n'.join(context_parts)

    def _format_rule(self, rule: SyntaxRule) -> str:
        """Format a rule for context injection."""
        parts = [
            f"## {rule.title}",
            f"*{rule.description}*" if rule.description else "",
            f"**Pattern:**\n```{rule.language}\n{rule.pattern}\n```" if rule.pattern else "",
        ]
        return '\n'.join(p for p in parts if p)

    def add_rule(self, rule: SyntaxRule):
        """Add a rule programmatically."""
        self.rules_by_id[rule.id] = rule
        self.rules_by_category[rule.category].append(rule)
        self.rules_by_language[rule.language].append(rule)
        for tag in rule.tags:
            self.rules_by_tag[tag].append(rule)

    def save_index(self, output_path: str):
        """Save indexed rules to JSON."""
        data = {
            'files': {k: asdict(v) for k, v in self.files.items()},
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_stats(self) -> dict[str, Any]:
        """Get hub statistics."""
        return {
            'total_files': len(self.files),
            'total_rules': len(self.rules_by_id),
            'by_language': {k: len(v) for k, v in self.rules_by_language.items()},
            'by_category': {k: len(v) for k, v in self.rules_by_category.items()},
            'by_tag': {k: len(v) for k, v in self.rules_by_tag.items()},
        }


def create_syntax_hub(rules_dir: str | None = None) -> SyntaxHub:
    """Create and initialize SyntaxHub."""
    hub = SyntaxHub(rules_dir)
    hub.initialize(create_defaults=True)
    return hub


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        hub = SyntaxHub(tmpdir)
        hub.initialize(create_defaults=True)

        stats = hub.get_stats()

        # Test getting rules
        python_rules = hub.get_rules(language='python', max_rules=5)
        for _r in python_rules:
            pass

        # Test search
        results = hub.search_rules('print', max_rules=5)
        for _r in results:
            pass

        # Test context generation
        context = hub.get_context_for_task("read a file and print its contents", language='psudo', max_tokens=500)


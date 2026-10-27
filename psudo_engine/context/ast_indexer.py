"""
AST Indexer for psudo-Engine Context Subsystem.

Parses repository code using tree-sitter to extract function signatures,
type definitions, and structural maps for context retrieval.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from tree_sitter import Language, Node, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = Parser = Node = None


@dataclass
class FunctionSignature:
    """Extracted function signature information."""
    name: str
    params: list[dict[str, str]]  # [{name, type}]
    return_type: str
    docstring: str
    file_path: str
    line_start: int
    line_end: int
    is_async: bool = False
    is_method: bool = False
    class_name: str | None = None


@dataclass
class TypeDefinition:
    """Extracted type definition (class, struct, enum, etc.)."""
    name: str
    kind: str  # class, struct, enum, interface, type_alias
    fields: list[dict[str, str]]  # [{name, type}]
    methods: list[str]
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""


@dataclass
class ImportInfo:
    """Extracted import information."""
    module: str
    names: list[str]
    file_path: str
    line: int


@dataclass
class FileIndex:
    """Complete index for a single file."""
    file_path: str
    language: str
    functions: list[FunctionSignature] = field(default_factory=list)
    types: list[TypeDefinition] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    docstring: str = ""


class ASTIndexer:
    """
    Tree-sitter based AST indexer for Python and Rust.

    Extracts structural information for context retrieval:
    - Function signatures with types
    - Type definitions (classes, structs, enums)
    - Imports/exports
    - Module-level docstrings
    """

    # Tree-sitter query strings for Python
    PYTHON_QUERIES = {
        'functions': """
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                return_type: (type)? @return_type
                body: (block (expression_statement (string))? @docstring)
            ) @function
        """,
        'classes': """
            (class_definition
                name: (identifier) @name
                body: (block) @body
            ) @class
        """,
        'imports': """
            (import_statement
                name: (dotted_name) @module
            ) @import
            (import_from_statement
                module_name: (dotted_name)? @module
                name: (import_list) @names
            ) @import_from
        """,
    }

    # Tree-sitter query strings for Rust
    RUST_QUERIES = {
        'functions': """
            (function_item
                name: (identifier) @name
                parameters: (parameters) @params
                return_type: (type)? @return_type
                body: (block) @body
            ) @function
        """,
        'structs': """
            (struct_item
                name: (type_identifier) @name
                body: (field_declaration_list) @fields
            ) @struct
        """,
        'enums': """
            (enum_item
                name: (type_identifier) @name
                body: (enum_variant_list) @variants
            ) @enum
        """,
        'impls': """
            (impl_item
                type: (type_identifier) @type
                body: (declaration_list) @methods
            ) @impl
        """,
        'imports': """
            (use_declaration
                argument: (use_tree) @tree
            ) @use
        """,
    }

    def __init__(self, repo_root: str, languages: list[str] | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.languages = languages or ['python', 'rust']
        self.parsers: dict[str, Parser] = {}
        self._init_parsers()

        self.file_index: dict[str, FileIndex] = {}
        self.function_index: dict[str, list[FunctionSignature]] = defaultdict(list)
        self.type_index: dict[str, list[TypeDefinition]] = defaultdict(list)

    def _init_parsers(self):
        """Initialize tree-sitter parsers for each language."""
        if not TREE_SITTER_AVAILABLE:
            return

        for lang in self.languages:
            parser = Parser()
            # Note: In practice, you'd load the actual language libraries
            # parser.set_language(Language('build/my-languages.so', lang))
            self.parsers[lang] = parser

    def index_repository(self, extensions: list[str] | None = None) -> dict[str, FileIndex]:
        """
        Index all files in the repository.

        Args:
            extensions: File extensions to index (default: .py, .rs)

        Returns:
            Dict of file_path -> FileIndex
        """
        if extensions is None:
            extensions = ['.py', '.rs']

        files = []
        for ext in extensions:
            files.extend(self.repo_root.rglob(f'*{ext}'))

        for file_path in files:
            if self._should_skip(file_path):
                continue

            rel_path = file_path.relative_to(self.repo_root)
            self._index_file(rel_path)

        return self.file_index

    def _should_skip(self, path: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {'.git', '__pycache__', 'target', 'node_modules', '.venv', 'venv', 'env'}
        return any(part in skip_dirs for part in path.parts)

    def _index_file(self, rel_path: Path):
        """Index a single file."""
        full_path = self.repo_root / rel_path

        try:
            content = full_path.read_text(encoding='utf-8')
        except Exception:
            return

        language = self._detect_language(rel_path)
        if language not in self.parsers and TREE_SITTER_AVAILABLE:
            return

        if TREE_SITTER_AVAILABLE and language in self.parsers:
            index = self._parse_with_tree_sitter(content, rel_path, language)
        else:
            index = self._parse_with_regex(content, rel_path, language)

        self.file_index[str(rel_path)] = index

        # Update global indexes
        for func in index.functions:
            self.function_index[func.name].append(func)
        for typ in index.types:
            self.type_index[typ.name].append(typ)

    def _detect_language(self, path: Path) -> str:
        """Detect language from file extension."""
        suffix = path.suffix.lower()
        if suffix == '.py':
            return 'python'
        elif suffix == '.rs':
            return 'rust'
        return 'unknown'

    def _parse_with_tree_sitter(self, content: str, rel_path: Path, language: str) -> FileIndex:
        """Parse using tree-sitter."""
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, 'utf-8'))

        index = FileIndex(
            file_path=str(rel_path),
            language=language,
            docstring=self._extract_module_docstring(content, language)
        )

        # Extract functions
        for func_node in self._query_nodes(tree, language, 'functions'):
            func = self._extract_function(func_node, content, rel_path, language)
            if func:
                index.functions.append(func)

        # Extract types
        for type_node in self._query_nodes(tree, language, 'classes' if language == 'python' else 'structs'):
            typ = self._extract_type(type_node, content, rel_path, language)
            if typ:
                index.types.append(typ)

        # Extract imports
        for import_node in self._query_nodes(tree, language, 'imports'):
            imp = self._extract_import(import_node, content, rel_path, language)
            if imp:
                index.imports.append(imp)

        return index

    def _query_nodes(self, tree, language: str, query_name: str):
        """Query tree-sitter for nodes matching pattern."""
        queries = self.PYTHON_QUERIES if language == 'python' else self.RUST_QUERIES
        query_str = queries.get(query_name, '')
        if not query_str:
            return []

        # Note: Actual tree-sitter query execution would go here
        # For now, return empty list
        return []

    def _parse_with_regex(self, content: str, rel_path: Path, language: str) -> FileIndex:
        """Fallback regex-based parsing when tree-sitter unavailable."""

        index = FileIndex(
            file_path=str(rel_path),
            language=language,
            docstring=self._extract_module_docstring(content, language)
        )

        if language == 'python':
            index.functions = self._extract_python_functions_regex(content, rel_path)
            index.types = self._extract_python_classes_regex(content, rel_path)
            index.imports = self._extract_python_imports_regex(content, rel_path)
        elif language == 'rust':
            index.functions = self._extract_rust_functions_regex(content, rel_path)
            index.types = self._extract_rust_types_regex(content, rel_path)
            index.imports = self._extract_rust_imports_regex(content, rel_path)

        return index

    def _extract_module_docstring(self, content: str, language: str) -> str:
        """Extract module-level docstring."""
        if language == 'python':
            import re
            match = re.match(r'^\s*("""|\'\'\')(.+?)\1', content, re.DOTALL)
            if match:
                return match.group(2).strip()
        elif language == 'rust':
            import re
            # Rust outer doc comments //! or /*! ... */
            lines = content.split('\n')
            doc_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('//!') or stripped.startswith('/*!'):
                    doc_lines.append(stripped[3:].strip())
                elif doc_lines and not stripped.startswith(('//!', '/*!', '*')):
                    break
            return '\n'.join(doc_lines).strip()
        return ""

    def _extract_python_functions_regex(self, content: str, rel_path: Path) -> list[FunctionSignature]:
        """Extract Python function signatures using regex."""
        import re

        functions = []
        # Pattern for function definitions
        pattern = re.compile(
            r'^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)'
            r'(?:\s*->\s*(?P<return_type>[^:]+))?\s*:',
            re.MULTILINE
        )

        lines = content.split('\n')
        for match in pattern.finditer(content):
            name = match.group('name')
            params_str = match.group('params').strip()
            return_type = match.group('return_type').strip() if match.group('return_type') else 'Any'
            line_start = content[:match.start()].count('\n')

            # Parse parameters
            params = []
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        parts = param.split(':')
                        p_name = parts[0].strip().lstrip('*')
                        p_type = parts[1].strip() if len(parts) > 1 else 'Any'
                        params.append({'name': p_name, 'type': p_type})

            # Find docstring
            docstring = ""
            body_start = line_start + 1
            if body_start < len(lines):
                next_line = lines[body_start].strip()
                if next_line.startswith(('"""', "'''")):
                    # Simple single-line docstring
                    docstring = next_line.strip('"\'')

            # Find function end (by indentation)
            indent = match.group('indent')
            line_end = line_start
            for i in range(line_start + 1, len(lines)):
                if lines[i].strip() and not lines[i].startswith(indent + ' ') and not lines[i].startswith('\t'):
                    break
                line_end = i

            functions.append(FunctionSignature(
                name=name,
                params=params,
                return_type=return_type,
                docstring=docstring,
                file_path=str(rel_path),
                line_start=line_start,
                line_end=line_end,
                is_async='async' in match.group(0)[:match.group(0).index('def')],
            ))

        return functions

    def _extract_python_classes_regex(self, content: str, rel_path: Path) -> list[TypeDefinition]:
        """Extract Python class definitions using regex."""
        import re

        types = []
        pattern = re.compile(
            r'^(?P<indent>\s*)class\s+(?P<name>\w+)(?:\((?P<bases>[^)]*)\))?\s*:',
            re.MULTILINE
        )

        lines = content.split('\n')
        for match in pattern.finditer(content):
            name = match.group('name')
            match.group('bases') or ''
            line_start = content[:match.start()].count('\n')

            # Extract methods (simplified)
            indent = match.group('indent')
            methods = []
            fields = []
            line_end = line_start
            in_class = False
            class_indent = None

            for i in range(line_start + 1, len(lines)):
                line = lines[i]
                if not in_class:
                    if line.strip() and (line.startswith(indent + ' ') or line.startswith('\t')):
                        in_class = True
                        class_indent = line[:len(line) - len(line.lstrip())]
                    else:
                        break

                if in_class:
                    stripped = line.strip()
                    if stripped and class_indent and not line.startswith(class_indent + ' '):
                        break

                    # Look for method definitions
                    method_match = re.match(r'\s+def\s+(\w+)', line)
                    if method_match:
                        methods.append(method_match.group(1))

                    # Look for field assignments (simplified)
                    field_match = re.match(r'\s+(\w+)\s*[:=]', line)
                    if field_match and not method_match:
                        field_name = field_match.group(1)
                        if not field_name.startswith('_'):
                            fields.append({'name': field_name, 'type': 'Any'})

                    line_end = i

            docstring = ""
            if line_start + 1 < len(lines):
                next_line = lines[line_start + 1].strip()
                if next_line.startswith(('"""', "'''")):
                    docstring = next_line.strip('"\'')

            types.append(TypeDefinition(
                name=name,
                kind='class',
                fields=fields,
                methods=methods,
                file_path=str(rel_path),
                line_start=line_start,
                line_end=line_end,
                docstring=docstring
            ))

        return types

    def _extract_python_imports_regex(self, content: str, rel_path: Path) -> list[ImportInfo]:
        """Extract Python imports using regex."""

        imports = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import '):
                module = stripped[7:].strip()
                for name in module.split(','):
                    imports.append(ImportInfo(
                        module=name.strip(),
                        names=[],
                        file_path=str(rel_path),
                        line=i
                    ))
            elif stripped.startswith('from ') and ' import ' in stripped:
                parts = stripped[5:].split(' import ')
                module = parts[0].strip()
                names = [n.strip() for n in parts[1].split(',')]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    file_path=str(rel_path),
                    line=i
                ))

        return imports

    def _extract_rust_functions_regex(self, content: str, rel_path: Path) -> list[FunctionSignature]:
        """Extract Rust function signatures using regex."""
        import re

        functions = []
        # Simplified pattern for Rust functions
        pattern = re.compile(
            r'(?:pub\s+)?(?:async\s+)?fn\s+(?P<name>\w+)\s*'
            r'(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)'
            r'(?:\s*->\s*(?P<return_type>[^{;]+))?\s*\{',
            re.MULTILINE | re.DOTALL
        )

        content.split('\n')
        for match in pattern.finditer(content):
            name = match.group('name')
            params_str = match.group('params').strip()
            return_type = match.group('return_type').strip() if match.group('return_type') else '()'
            line_start = content[:match.start()].count('\n')

            params = []
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    if param and param != '&self' and param != '&mut self' and param != 'self':
                        parts = param.split(':')
                        p_name = parts[0].strip().lstrip('&mut ').lstrip('&')
                        p_type = parts[1].strip() if len(parts) > 1 else 'Unknown'
                        params.append({'name': p_name, 'type': p_type})

            functions.append(FunctionSignature(
                name=name,
                params=params,
                return_type=return_type,
                docstring="",  # Rust doc comments would need separate extraction
                file_path=str(rel_path),
                line_start=line_start,
                line_end=line_start + 10,  # Approximate
                is_async='async' in match.group(0)[:match.group(0).index('fn')],
            ))

        return functions

    def _extract_rust_types_regex(self, content: str, rel_path: Path) -> list[TypeDefinition]:
        """Extract Rust struct/enum definitions using regex."""
        import re

        types = []
        content.split('\n')

        # Structs
        struct_pattern = re.compile(r'(?:pub\s+)?struct\s+(?P<name>\w+)(?:\s*<[^>]*>)?\s*(?:\{|\()')
        for match in struct_pattern.finditer(content):
            name = match.group('name')
            line_start = content[:match.start()].count('\n')
            types.append(TypeDefinition(
                name=name,
                kind='struct',
                fields=[],  # Would need more parsing
                methods=[],
                file_path=str(rel_path),
                line_start=line_start,
                line_end=line_start + 5,
            ))

        # Enums
        enum_pattern = re.compile(r'(?:pub\s+)?enum\s+(?P<name>\w+)(?:\s*<[^>]*>)?\s*\{')
        for match in enum_pattern.finditer(content):
            name = match.group('name')
            line_start = content[:match.start()].count('\n')
            types.append(TypeDefinition(
                name=name,
                kind='enum',
                fields=[],
                methods=[],
                file_path=str(rel_path),
                line_start=line_start,
                line_end=line_start + 5,
            ))

        return types

    def _extract_rust_imports_regex(self, content: str, rel_path: Path) -> list[ImportInfo]:
        """Extract Rust use declarations using regex."""
        import re

        imports = []
        pattern = re.compile(r'use\s+([^;]+);')
        lines = content.split('\n')

        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                module = match.group(1).strip()
                imports.append(ImportInfo(
                    module=module,
                    names=[],
                    file_path=str(rel_path),
                    line=i
                ))

        return imports

    # Placeholder methods for tree-sitter extraction
    def _extract_function(self, node: Any, content: str, rel_path: Path, language: str) -> FunctionSignature | None:
        return None

    def _extract_type(self, node: Any, content: str, rel_path: Path, language: str) -> TypeDefinition | None:
        return None

    def _extract_import(self, node: Any, content: str, rel_path: Path, language: str) -> ImportInfo | None:
        return None

    def get_functions_by_name(self, name: str) -> list[FunctionSignature]:
        """Get all functions with a given name."""
        return self.function_index.get(name, [])

    def get_types_by_name(self, name: str) -> list[TypeDefinition]:
        """Get all types with a given name."""
        return self.type_index.get(name, [])

    def search_functions(self, query: str) -> list[FunctionSignature]:
        """Search functions by name substring."""
        results = []
        query_lower = query.lower()
        for funcs in self.function_index.values():
            for func in funcs:
                if query_lower in func.name.lower() or query_lower in func.docstring.lower():
                    results.append(func)
        return results

    def get_file_index(self, file_path: str) -> FileIndex | None:
        """Get index for a specific file."""
        return self.file_index.get(file_path)

    def save_index(self, output_path: str):
        """Save index to JSON file."""
        data = {
            'files': {k: asdict(v) for k, v in self.file_index.items()},
            'functions': {k: [asdict(f) for f in v] for k, v in self.function_index.items()},
            'types': {k: [asdict(t) for t in v] for k, v in self.type_index.items()},
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_index(self, input_path: str):
        """Load index from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        self.file_index = {k: FileIndex(**v) for k, v in data.get('files', {}).items()}
        self.function_index = defaultdict(list, {
            k: [FunctionSignature(**f) for f in v] for k, v in data.get('functions', {}).items()
        })
        self.type_index = defaultdict(list, {
            k: [TypeDefinition(**t) for t in v] for k, v in data.get('types', {}).items()
        })


def create_ast_indexer(repo_root: str, config: dict | None = None) -> ASTIndexer:
    """Create ASTIndexer from config."""
    config = config or {}
    languages = config.get('languages', ['python', 'rust'])
    return ASTIndexer(repo_root, languages)


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test Python file
        test_file = Path(tmpdir) / "test_module.py"
        test_file.write_text("""
\"\"\"Test module docstring.\"\"\"

import os
from typing import List, Optional

def hello_world(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

async def fetch_data(url: str) -> dict:
    \"\"\"Fetch data from URL.\"\"\"
    return {"url": url}

class DataProcessor:
    \"\"\"Process data.\"\"\"

    def __init__(self, config: dict):
        self.config = config

    def process(self, items: List[str]) -> List[str]:
        return [item.upper() for item in items]
""")

        # Create test Rust file
        test_rs = Path(tmpdir) / "test_lib.rs"
        test_rs.write_text("""
//! Test library docstring.

use std::collections::HashMap;

pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

pub struct Config {
    pub debug: bool,
    pub timeout: u32,
}

impl Config {
    pub fn new() -> Self {
        Config { debug: false, timeout: 30 }
    }
}

pub enum Status {
    Ok,
    Error(String),
}
""")

        indexer = ASTIndexer(tmpdir)
        indexer.index_repository()


        for _path, _idx in indexer.file_index.items():
            pass


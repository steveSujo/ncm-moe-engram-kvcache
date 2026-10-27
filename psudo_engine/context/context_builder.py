"""
Context Builder for psudo-Engine Context Subsystem.

Assembles retrieved syntax rules, AST signatures, and other context
into a token-budgeted prompt for the model.
"""

from dataclasses import dataclass, field
from typing import Any

from .ast_indexer import ASTIndexer, FunctionSignature, TypeDefinition
from .retriever import HybridRetriever, SyntaxRuleRetriever
from .syntax_hub import SyntaxHub, SyntaxRule


@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    content: str
    source: str  # 'ast', 'syntax', 'retrieval', 'task'
    priority: int
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_tokens: int = 1500
    max_syntax_rules: int = 10
    max_ast_functions: int = 5
    max_ast_types: int = 3
    include_docstrings: bool = True
    include_patterns: bool = True
    syntax_priority_weight: float = 1.0
    ast_priority_weight: float = 1.2
    retrieval_priority_weight: float = 1.5


class TokenEstimator:
    """Estimate token counts for text."""

    # Rough approximation: 1 token ≈ 4 characters for English/code
    CHARS_PER_TOKEN = 4

    @classmethod
    def estimate(cls, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // cls.CHARS_PER_TOKEN)

    @classmethod
    def estimate_chunks(cls, chunks: list[ContextChunk]) -> int:
        """Estimate total tokens for chunks."""
        return sum(chunk.tokens for chunk in chunks)


class ContextBuilder:
    """
    Builds token-budgeted context from multiple sources.

    Sources:
    - Task description
    - Retrieved syntax rules (from SyntaxHub + Retriever)
    - AST function signatures (from ASTIndexer)
    - AST type definitions
    - Relevant code patterns
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        syntax_hub: SyntaxHub | None = None,
        ast_indexer: ASTIndexer | None = None,
        retriever: HybridRetriever | None = None
    ):
        self.config = config or ContextConfig()
        self.syntax_hub = syntax_hub
        self.ast_indexer = ast_indexer
        self.retriever = retriever

        if syntax_hub and retriever:
            self.syntax_retriever = SyntaxRuleRetriever(syntax_hub)
            self.syntax_retriever.retriever = retriever
        else:
            self.syntax_retriever = None

    def build_context(
        self,
        task: str,
        language: str = 'psudo',
        relevant_files: list[str] | None = None,
        additional_context: str | None = None
    ) -> str:
        """
        Build complete context for a task.

        Args:
            task: Task description
            language: Target language
            relevant_files: List of file paths to include AST context from
            additional_context: Additional context string to include

        Returns:
            Formatted context string within token budget
        """
        chunks = []

        # 1. Task description (highest priority)
        task_chunk = self._create_task_chunk(task)
        chunks.append(task_chunk)

        # 2. Additional context if provided
        if additional_context:
            chunks.append(ContextChunk(
                content=additional_context,
                source='additional',
                priority=100,
                tokens=TokenEstimator.estimate(additional_context)
            ))

        # 3. Syntax rules relevant to task
        if self.syntax_hub:
            syntax_chunks = self._create_syntax_chunks(task, language)
            chunks.extend(syntax_chunks)

        # 4. AST context from relevant files
        if self.ast_indexer and relevant_files:
            ast_chunks = self._create_ast_chunks(relevant_files, language)
            chunks.extend(ast_chunks)

        # 5. Retrieved context from retriever
        if self.retriever:
            retrieval_chunks = self._create_retrieval_chunks(task, language)
            chunks.extend(retrieval_chunks)

        # Select and format within budget
        return self._select_and_format(chunks)

    def _create_task_chunk(self, task: str) -> ContextChunk:
        """Create context chunk for task description."""
        content = f"# Task\n{task}"
        return ContextChunk(
            content=content,
            source='task',
            priority=200,
            tokens=TokenEstimator.estimate(content),
            metadata={'task': task}
        )

    def _create_syntax_chunks(
        self,
        task: str,
        language: str
    ) -> list[ContextChunk]:
        """Create context chunks from syntax rules."""
        chunks = []

        # Get rules from syntax hub
        rules = self.syntax_hub.get_rules(
            language=language,
            max_rules=self.config.max_syntax_rules
        )

        # Also search for task-relevant rules
        if self.syntax_retriever and self.syntax_retriever._indexed:
            search_results = self.syntax_retriever.retrieve_for_task(
                task, language=language, top_k=self.config.max_syntax_rules
            )
            # Merge and deduplicate
            seen = set()
            merged_rules = []
            for rule in rules:
                if rule.id not in seen:
                    seen.add(rule.id)
                    merged_rules.append((1.0, rule))
            for score, rule in search_results:
                if rule.id not in seen:
                    seen.add(rule.id)
                    merged_rules.append((score, rule))

            merged_rules.sort(key=lambda x: -x[0])
            rules = [r for _, r in merged_rules[:self.config.max_syntax_rules]]

        for rule in rules:
            content = self._format_syntax_rule(rule)
            tokens = TokenEstimator.estimate(content)
            priority = int(rule.priority * 10 * self.config.syntax_priority_weight)

            chunks.append(ContextChunk(
                content=content,
                source='syntax',
                priority=priority,
                tokens=tokens,
                metadata={
                    'rule_id': rule.id,
                    'category': rule.category,
                    'language': rule.language,
                }
            ))

        return chunks

    def _format_syntax_rule(self, rule: SyntaxRule) -> str:
        """Format a syntax rule for context."""
        parts = [f"## {rule.title}"]

        if rule.description and self.config.include_docstrings:
            parts.append(f"*{rule.description}*")

        if rule.pattern and self.config.include_patterns:
            parts.append(f"**Pattern:**\n```{rule.language}\n{rule.pattern}\n```")

        if rule.tags:
            parts.append(f"**Tags:** {', '.join(rule.tags)}")

        return '\n\n'.join(parts)

    def _create_ast_chunks(
        self,
        file_paths: list[str],
        language: str
    ) -> list[ContextChunk]:
        """Create context chunks from AST index."""
        chunks = []

        for file_path in file_paths:
            file_index = self.ast_indexer.get_file_index(file_path)
            if not file_index:
                continue

            # File overview
            if file_index.docstring:
                content = f"# File: {file_path}\n{file_index.docstring}"
                chunks.append(ContextChunk(
                    content=content,
                    source='ast',
                    priority=50,
                    tokens=TokenEstimator.estimate(content),
                    metadata={'file': file_path, 'type': 'docstring'}
                ))

            # Functions
            for func in file_index.functions[:self.config.max_ast_functions]:
                content = self._format_function_signature(func)
                tokens = TokenEstimator.estimate(content)
                priority = 60

                chunks.append(ContextChunk(
                    content=content,
                    source='ast',
                    priority=priority,
                    tokens=tokens,
                    metadata={
                        'file': file_path,
                        'type': 'function',
                        'name': func.name,
                    }
                ))

            # Types
            for typ in file_index.types[:self.config.max_ast_types]:
                content = self._format_type_definition(typ)
                tokens = TokenEstimator.estimate(content)
                priority = 55

                chunks.append(ContextChunk(
                    content=content,
                    source='ast',
                    priority=priority,
                    tokens=tokens,
                    metadata={
                        'file': file_path,
                        'type': 'type_definition',
                        'name': typ.name,
                    }
                ))

        return chunks

    def _format_function_signature(self, func: FunctionSignature) -> str:
        """Format function signature for context."""
        params_str = ', '.join(f"{p['name']}: {p['type']}" for p in func.params)
        parts = [
            f"### Function: {func.name}",
            f"```{func.file_path.split('.')[-1] if '.' in func.file_path else 'python'}",
            f"{'async ' if func.is_async else ''}def {func.name}({params_str}) -> {func.return_type}:",
            "    ...",
            "```",
        ]
        if func.docstring and self.config.include_docstrings:
            parts.insert(1, f"*{func.docstring}*")
        return '\n'.join(parts)

    def _format_type_definition(self, typ: TypeDefinition) -> str:
        """Format type definition for context."""
        parts = [
            f"### {typ.kind.capitalize()}: {typ.name}",
        ]

        if typ.fields:
            fields_str = ', '.join(f"{f['name']}: {f['type']}" for f in typ.fields)
            parts.append(f"Fields: {fields_str}")

        if typ.methods:
            methods_str = ', '.join(typ.methods)
            parts.append(f"Methods: {methods_str}")

        if typ.docstring and self.config.include_docstrings:
            parts.insert(1, f"*{typ.docstring}*")

        return '\n'.join(parts)

    def _create_retrieval_chunks(
        self,
        task: str,
        language: str
    ) -> list[ContextChunk]:
        """Create context chunks from retriever."""
        chunks = []

        # Search for relevant documents
        results = self.retriever.search(task, top_k=10)

        for score, doc in results:
            # Filter by language if possible
            doc_lang = doc.metadata.get('language', 'generic')
            if doc_lang != language and doc_lang != 'generic':
                continue

            content = self._format_retrieval_doc(doc, score)
            tokens = TokenEstimator.estimate(content)
            priority = int(score * 100 * self.config.retrieval_priority_weight)

            chunks.append(ContextChunk(
                content=content,
                source='retrieval',
                priority=priority,
                tokens=tokens,
                metadata={
                    'doc_id': doc.id,
                    'score': score,
                    'metadata': doc.metadata,
                }
            ))

        return chunks

    def _format_retrieval_doc(self, doc: Any, score: float) -> str:
        """Format retrieved document for context."""
        return f"**Retrieved (score: {score:.3f}):**\n```\n{doc.content}\n```"

    def _select_and_format(self, chunks: list[ContextChunk]) -> str:
        """Select chunks within token budget and format."""
        # Sort by priority (descending)
        chunks.sort(key=lambda c: -c.priority)

        selected = []
        total_tokens = 0

        for chunk in chunks:
            if total_tokens + chunk.tokens <= self.config.max_tokens:
                selected.append(chunk)
                total_tokens += chunk.tokens
            else:
                # Try to fit partial content if it's a high-priority chunk
                if chunk.priority > 100 and chunk.source == 'task':
                    # Always include task
                    remaining = self.config.max_tokens - total_tokens
                    if remaining > 50:
                        truncated = chunk.content[:remaining * 4]
                        selected.append(ContextChunk(
                            content=truncated + "... [truncated]",
                            source=chunk.source,
                            priority=chunk.priority,
                            tokens=remaining,
                            metadata=chunk.metadata
                        ))
                        total_tokens += remaining
                break

        # Format selected chunks
        parts = []
        for chunk in selected:
            parts.append(chunk.content)

        result = '\n\n---\n\n'.join(parts)

        # Add token usage info
        result += f"\n\n<!-- Context tokens: {total_tokens}/{self.config.max_tokens} -->"

        return result

    def build_minimal_context(self, task: str, language: str = 'psudo') -> str:
        """Build minimal context with just task and essential syntax."""
        chunks = []

        # Task
        chunks.append(self._create_task_chunk(task))

        # Essential syntax rules only (high priority)
        if self.syntax_hub:
            rules = self.syntax_hub.get_rules(
                language=language,
                max_rules=5
            )
            for rule in rules:
                if rule.priority >= 8:
                    content = self._format_syntax_rule(rule)
                    chunks.append(ContextChunk(
                        content=content,
                        source='syntax',
                        priority=int(rule.priority * 10),
                        tokens=TokenEstimator.estimate(content),
                        metadata={'rule_id': rule.id}
                    ))

        return self._select_and_format(chunks)


def create_context_builder(
    config: dict | None = None,
    syntax_hub: SyntaxHub | None = None,
    ast_indexer: ASTIndexer | None = None,
    retriever: HybridRetriever | None = None
) -> ContextBuilder:
    """Create ContextBuilder from config."""
    cfg = config or {}
    context_config = ContextConfig(
        max_tokens=cfg.get('max_tokens', 1500),
        max_syntax_rules=cfg.get('max_syntax_rules', 10),
        max_ast_functions=cfg.get('max_ast_functions', 5),
        max_ast_types=cfg.get('max_ast_types', 3),
    )
    return ContextBuilder(context_config, syntax_hub, ast_indexer, retriever)


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create components
        syntax_hub = SyntaxHub(tmpdir)
        syntax_hub.initialize(create_defaults=True)

        retriever = HybridRetriever()

        # Add some test documents to retriever
        retriever.add_document(
            "test_1",
            "def read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()",
            {"language": "python", "type": "builtin"}
        )
        retriever.build()

        # Create context builder
        builder = create_context_builder(
            config={'max_tokens': 500},
            syntax_hub=syntax_hub,
            retriever=retriever
        )

        # Build context
        context = builder.build_context(
            task="Read a file and print its contents",
            language='psudo'
        )



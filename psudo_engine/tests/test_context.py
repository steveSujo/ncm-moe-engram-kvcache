"""
Unit tests for Context & Retrieval Subsystem.
"""

import tempfile
from pathlib import Path
import pytest

from psudo_engine.context.ast_indexer import ASTIndexer
from psudo_engine.context.syntax_hub import SyntaxHub
from psudo_engine.context.retriever import HybridRetriever
from psudo_engine.context.context_builder import ContextBuilder, ContextConfig


class TestASTIndexer:
    """Tests for ASTIndexer."""
    
    def test_index_python_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test Python file
            test_file = Path(tmpdir) / "test_module.py"
            test_file.write_text("""
def hello(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

class DataProcessor:
    \"\"\"Process data.\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
    
    def process(self, items: list[str]) -> list[str]:
        return [item.upper() for item in items]
""")
            
            indexer = ASTIndexer(tmpdir)
            indexer.index_repository()
            
            # Check file indexed
            assert str(test_file.relative_to(tmpdir)) in indexer.file_index
            
            file_index = indexer.get_file_index(str(test_file.relative_to(tmpdir)))
            assert file_index is not None
            assert len(file_index.functions) >= 2
            assert len(file_index.types) >= 1
    
    def test_search_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("""
def process_data(data: list) -> dict:
    return {"processed": True}

def validate_input(input: str) -> bool:
    return len(input) > 0
""")
            
            indexer = ASTIndexer(tmpdir)
            indexer.index_repository()
            
            results = indexer.search_functions("process")
            assert len(results) >= 1
            assert any(r.name == "process_data" for r in results)


class TestSyntaxHub:
    """Tests for SyntaxHub."""
    
    def test_load_default_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = SyntaxHub(tmpdir)
            hub.initialize(create_defaults=True)
            
            stats = hub.get_stats()
            assert stats['total_rules'] > 0
            assert 'python' in stats['by_language']
            assert 'rust' in stats['by_language']
            assert 'psudo' in stats['by_language']
    
    def test_get_rules_by_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = SyntaxHub(tmpdir)
            hub.initialize(create_defaults=True)
            
            python_rules = hub.get_rules(language='python')
            assert len(python_rules) > 0
            assert all(r.language == 'python' for r in python_rules)
    
    def test_search_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = SyntaxHub(tmpdir)
            hub.initialize(create_defaults=True)
            
            results = hub.search_rules("print")
            assert len(results) > 0
            assert any("print" in r.title.lower() for r in results)
    
    def test_get_context_for_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = SyntaxHub(tmpdir)
            hub.initialize(create_defaults=True)
            
            context = hub.get_context_for_task(
                "read a file and print its contents",
                language='psudo',
                max_tokens=500
            )
            
            assert len(context) > 0
            assert "read_file" in context or "print" in context


class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    def test_add_and_search_documents(self):
        retriever = HybridRetriever(
            bm25_weight=0.5,
            embedding_weight=0.5
        )
        
        docs = [
            ("doc1", "def read_file(path: str) -> str:\n    return open(path).read()", 
             {"language": "python", "type": "builtin"}),
            ("doc2", "fn read_file(path: &str) -> String {\n    std::fs::read_to_string(path)\n}",
             {"language": "rust", "type": "builtin"}),
            ("doc3", "read_file(path: string) -> string",
             {"language": "psudo", "type": "builtin"}),
        ]
        
        retriever.add_documents(docs)
        retriever.build()
        
        results = retriever.search("read file", top_k=2)
        assert len(results) == 2
        
        for score, doc in results:
            assert score > 0
            assert "read" in doc.content.lower()
    
    def test_bm25_only_search(self):
        retriever = HybridRetriever(bm25_weight=1.0, embedding_weight=0.0)
        
        retriever.add_document("doc1", "function hello world", {"lang": "python"})
        retriever.add_document("doc2", "function goodbye world", {"lang": "python"})
        retriever.build()
        
        results = retriever.search_bm25_only("hello", top_k=1)
        assert len(results) == 1
        assert results[0][1].id == "doc1"
    
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever()
            retriever.add_document("doc1", "test content", {"tag": "test"})
            retriever.build()
            
            save_path = Path(tmpdir) / "retriever.pkl"
            retriever.save(str(save_path))
            
            loaded = HybridRetriever.load(str(save_path))
            assert len(loaded.documents) == 1
            assert loaded.documents[0].content == "test content"


class TestContextBuilder:
    """Tests for ContextBuilder."""
    
    def test_build_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup components
            syntax_hub = SyntaxHub(tmpdir)
            syntax_hub.initialize(create_defaults=True)
            
            retriever = HybridRetriever()
            retriever.add_document(
                "rule1", "def read_file(path: str) -> str:\n    return open(path).read()",
                {"language": "python", "type": "builtin"}
            )
            retriever.build()
            
            # Create context builder
            config = ContextConfig(max_tokens=500)
            builder = ContextBuilder(
                config=config,
                syntax_hub=syntax_hub,
                retriever=retriever
            )
            
            context = builder.build_context(
                task="Read a file and print its contents",
                language='psudo'
            )
            
            assert len(context) > 0
            assert "Task" in context
            # Should include syntax rules
            assert "read_file" in context or "print" in context
    
    def test_minimal_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            syntax_hub = SyntaxHub(tmpdir)
            syntax_hub.initialize(create_defaults=True)
            
            builder = ContextBuilder(config=ContextConfig(max_tokens=300), syntax_hub=syntax_hub)
            
            context = builder.build_minimal_context(
                task="Print hello world",
                language='psudo'
            )
            
            assert len(context) > 0
            # Should be within token budget
            assert len(context) < 300 * 4  # chars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

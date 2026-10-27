"""
Tests for Active Learning Module
=================================

Validates:
- PlasticityController LR scheduling
- ActiveLearner conversation learning loop
- Gradient accumulation and updates
- Evaluation metrics
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.active_learning import (
    ActiveLearner,
    ConversationTurn,
    PlasticityConfig,
    PlasticityController,
    create_active_learner,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.active_learning import (
    PlasticityController,
    ActiveLearner,
    PlasticityConfig,
    ConversationTurn,
    create_active_learner
)


def get_test_model():
    """Create a simple test model with wte for ActiveLearner."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(1000, 128)
            self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.lm_head = nn.Linear(128, 1000)
        
        def forward(self, tokens, level=None):
            x = self.wte(tokens)
            x = self.layers(x)
            logits = self.lm_head(x)
            # Return dummy loss for testing
            loss = logits.mean()
            return logits, loss
    
    return TestModel()


def get_test_tokenizer():
    """Create a mock tokenizer."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.eos_token_id = 999
            self.pad_token_id = 0
        
        def encode(self, text):
            return [hash(text) % 1000]
        
        def decode(self, tokens):
            return "mock"
    
    return MockTokenizer()


def test_plasticity_controller_creation():
    """Test PlasticityController creation."""
    model = get_test_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    config = PlasticityConfig(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-3,
        plasticity_decay=0.99,
        plasticity_boost_on_novelty=1.5,
        novelty_threshold=0.5,
        novelty_window=1000,
        newborn_rate=0.1,
        newborn_min_steps=100,
        eval_interval=10,
        eval_samples=10
    )
    
    controller = PlasticityController(config, optimizer)
    
    assert controller.config == config
    assert controller.current_plasticity == 1.0
    assert controller.step_count == 0
    
    print("✓ PlasticityController creation successful")


def test_plasticity_controller_step():
    """Test PlasticityController step updates."""
    model = get_test_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    config = PlasticityConfig(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-3,
        plasticity_decay=0.99,
        plasticity_boost_on_novelty=1.5,
        novelty_threshold=0.5,
        novelty_window=1000,
        newborn_rate=0.1,
        newborn_min_steps=100,
        eval_interval=10,
        eval_samples=10
    )
    
    controller = PlasticityController(config, optimizer)
    
    # Initial state
    initial_plasticity = controller.current_plasticity
    initial_step = controller.step_count
    
    # Step WITHOUT novelty (only decay)
    controller.step(loss=1.0)
    
    # Check updates
    assert controller.step_count == initial_step + 1
    assert controller.current_plasticity < initial_plasticity  # decayed
    
    print("✓ PlasticityController step successful")
    """Test PlasticityController evaluation step."""
    model = get_test_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    config = PlasticityConfig(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-3,
        plasticity_decay=0.99,
        plasticity_boost_on_novelty=1.5,
        novelty_threshold=0.5,
        novelty_window=1000,
        newborn_rate=0.1,
        newborn_min_steps=100,
        eval_interval=10,
        eval_samples=10
    )
    
    controller = PlasticityController(config, optimizer)
    
    # Run eval steps
    for i in range(15):
        controller.step(loss=1.0)
    
    # Should have decayed
    assert controller.current_plasticity < 1.0
    
    print("✓ PlasticityController evaluation successful")


def test_active_learner_creation():
    """Test ActiveLearner creation."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-3,
        plasticity_decay=0.99,
        plasticity_boost_on_novelty=1.5,
        novelty_threshold=0.5,
        novelty_window=1000,
        newborn_rate=0.1,
        newborn_min_steps=100,
        eval_interval=10,
        eval_samples=10
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    
    assert learner.model == model
    assert learner.tokenizer == tokenizer
    assert learner.config == config
    assert learner.optimizer == optimizer
    assert learner.plasticity is not None
    assert len(learner.conversation_history) == 0
    
    print("✓ ActiveLearner creation successful")


def test_active_learner_add_turn():
    """Test ActiveLearner adding conversation turns."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig()
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    
    # Add conversation turns
    for i in range(5):
        turn = ConversationTurn(
            user_input=f"User message {i}",
            model_response=f"Model response {i}",
            tokens=torch.randint(0, 1000, (1, 10)),
            level=torch.zeros(1, 10, dtype=torch.long),
            reward=0.8,
        )
        learner.add_turn(turn)
    
    assert len(learner.conversation_history) == 5
    assert len(learner.learning_buffer) == 5
    
    print("✓ ActiveLearner add_turn successful")


def test_active_learner_compute_novelty():
    """Test ActiveLearner compute_novelty."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig()
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    
    # Add some history
    for i in range(15):
        turn = ConversationTurn(
            user_input=f"User message {i}",
            model_response=f"Model response {i}",
            tokens=torch.randint(0, 1000, (1, 10)),
            level=torch.zeros(1, 10, dtype=torch.long),
            reward=0.8,
        )
        learner.add_turn(turn)
    
    # Compute novelty
    tokens = torch.randint(0, 1000, (1, 10))
    novelty = learner.compute_novelty(tokens)
    
    assert 0.0 <= novelty <= 1.0
    
    print(f"✓ ActiveLearner compute_novelty: {novelty:.4f}")


def test_active_learner_learn_from_buffer():
    """Test ActiveLearner learn_from_buffer."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-3,
        plasticity_decay=0.99,
        plasticity_boost_on_novelty=1.5,
        novelty_threshold=0.5,
        novelty_window=1000,
        newborn_rate=0.1,
        newborn_min_steps=100,
        eval_interval=10,
        eval_samples=10
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    learner.buffer_size = 4  # Small buffer for test
    
    # Add enough turns to trigger learning
    for i in range(6):
        turn = ConversationTurn(
            user_input=f"User message {i}",
            model_response=f"Model response {i}",
            tokens=torch.randint(0, 1000, (1, 10)),
            level=torch.zeros(1, 10, dtype=torch.long),
            reward=0.8,
        )
        learner.add_turn(turn)
    
    # Learn from buffer
    metrics = learner.learn_from_buffer()
    
    # Should have learned (buffer >= buffer_size)
    if metrics:
        assert "loss" in metrics
        assert "novelty" in metrics
        assert "plasticity" in metrics
        assert "lr" in metrics
    
    print("✓ ActiveLearner learn_from_buffer successful")


def test_active_learner_evaluate():
    """Test ActiveLearner evaluate."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig()
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    
    # Create eval data
    eval_data = []
    for i in range(5):
        turn = ConversationTurn(
            user_input=f"Eval {i}",
            model_response=f"Response {i}",
            tokens=torch.randint(0, 1000, (1, 10)),
            level=torch.zeros(1, 10, dtype=torch.long),
            reward=0.8,
        )
        eval_data.append(turn)
    
    metrics = learner.evaluate(eval_data)
    
    if metrics:
        assert "eval_loss" in metrics
        assert "eval_perplexity" in metrics
        assert isinstance(metrics["eval_loss"], float)
        assert isinstance(metrics["eval_perplexity"], float)
    
    print("✓ ActiveLearner evaluate successful")


def test_active_learner_checkpoint():
    """Test ActiveLearner checkpoint save/load."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    
    config = PlasticityConfig()
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    learner = ActiveLearner(model, tokenizer, plasticity_config=config, optimizer=optimizer)
    
    # Add some state
    turn = ConversationTurn(
        user_input="test",
        model_response="test",
        tokens=torch.randint(0, 1000, (1, 10)),
        level=torch.zeros(1, 10, dtype=torch.long),
        reward=1.0
    )
    learner.add_turn(turn)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
    
    try:
        # Save
        learner.save_checkpoint(checkpoint_path)
        
        # Create new learner and load
        model2 = get_test_model()
        tokenizer2 = get_test_tokenizer()
        optimizer2 = optim.AdamW(model2.parameters(), lr=config.base_lr)
        for group in optimizer2.param_groups:
            group['initial_lr'] = group['lr']
        
        learner2 = ActiveLearner(model2, tokenizer2, plasticity_config=config, optimizer=optimizer2)
        learner2.load_checkpoint(checkpoint_path)
        
        # Check state restored
        assert len(learner2.conversation_history) == 1
        
        print("✓ ActiveLearner checkpoint save/load successful")
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def test_conversation_turn():
    """Test ConversationTurn dataclass."""
    tokens = torch.randint(0, 1000, (1, 10))
    level = torch.zeros(1, 10, dtype=torch.long)
    
    turn = ConversationTurn(
        user_input="Hello",
        model_response="Hi there!",
        tokens=tokens,
        level=level,
        reward=0.9,
    )
    
    assert turn.user_input == "Hello"
    assert turn.model_response == "Hi there!"
    assert turn.reward == 0.9
    assert turn.timestamp >= 0.0
    
    print("✓ ConversationTurn successful")


def test_plasticity_config_defaults():
    """Test PlasticityConfig defaults."""
    config = PlasticityConfig()
    
    assert config.base_lr == 1e-4
    assert config.min_lr == 1e-6
    assert config.max_lr == 1e-3
    assert config.plasticity_decay == 0.999
    assert config.plasticity_boost_on_novelty == 1.5
    assert config.novelty_threshold == 0.5
    assert config.novelty_window == 1000
    assert config.newborn_rate == 0.01
    assert config.newborn_min_steps == 1000
    assert config.eval_interval == 100
    assert config.eval_samples == 50
    
    print("✓ PlasticityConfig defaults correct")


def test_create_active_learner_factory():
    """Test create_active_learner factory function."""
    model = get_test_model()
    tokenizer = get_test_tokenizer()
    config = PlasticityConfig()
    
    learner, optimizer = create_active_learner(
        model, tokenizer,
        lr_trunk=config.base_lr,
        lr_engram=config.base_lr * 0.3,
        lr_expert=config.base_lr * 10,
        device=torch.device('cpu')
    )
    
    assert isinstance(learner, ActiveLearner)
    assert isinstance(optimizer, optim.Optimizer)
    assert learner.optimizer == optimizer
    
    print("✓ create_active_learner factory successful")


if __name__ == "__main__":
    test_plasticity_controller_creation()
    test_plasticity_controller_step()
    test_plasticity_controller_eval()
    test_active_learner_creation()
    test_active_learner_add_turn()
    test_active_learner_compute_novelty()
    test_active_learner_learn_from_buffer()
    test_active_learner_evaluate()
    test_active_learner_checkpoint()
    test_conversation_turn()
    test_plasticity_config_defaults()
    test_create_active_learner_factory()
    print("\n=== All Active Learning tests passed! ===")
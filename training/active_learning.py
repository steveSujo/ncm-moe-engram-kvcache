"""
Active Learning Module (mini-AGI LiveLearner style)

Implements continuous learning from conversation with plasticity control.
Inspired by mini-AGI LiveLearner for online adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlasticityConfig:
    """Configuration for plasticity controller."""
    # Learning rate schedule
    base_lr: float = 1e-4
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    
    # Plasticity decay
    plasticity_decay: float = 0.999  # per step
    plasticity_boost_on_novelty: float = 1.5
    
    # Novelty detection
    novelty_threshold: float = 0.5  # cosine similarity threshold
    novelty_window: int = 1000  # steps to track novelty
    
    # Newborn rate (for new expert allocation)
    newborn_rate: float = 0.01  # fraction of experts to reinitialize
    newborn_min_steps: int = 1000
    
    # Evaluation
    eval_interval: int = 100
    eval_samples: int = 50


@dataclass
class ConversationTurn:
    """Single conversation turn for active learning."""
    user_input: str
    model_response: str
    tokens: torch.Tensor
    level: torch.Tensor
    reward: float = 0.0  # Optional human feedback
    timestamp: float = 0.0
    context: dict = field(default_factory=dict)


class PlasticityController:
    """
    Controls plasticity for active learning.
    Adjusts learning rates and newborn rates based on novelty and performance.
    """
    
    def __init__(self, config: PlasticityConfig, optimizer: torch.optim.Optimizer):
        self.config = config
        self.optimizer = optimizer
        self.step_count = 0
        
        # Plasticity state
        self.current_plasticity = 1.0
        self.novelty_buffer = deque(maxlen=config.novelty_window)
        self.performance_buffer = deque(maxlen=100)
        
        # Track LR multipliers per param group
        self.lr_multipliers = [1.0] * len(optimizer.param_groups)
        
        logger.info(f"PlasticityController initialized: base_lr={config.base_lr}, decay={config.plasticity_decay}")
    
    def step(self, novelty_score: float = None, loss: float = None):
        """Update plasticity state."""
        self.step_count += 1
        
        # Decay plasticity
        self.current_plasticity *= self.config.plasticity_decay
        
        # Boost on novelty
        if novelty_score is not None:
            self.novelty_buffer.append(novelty_score)
            if novelty_score > self.config.novelty_threshold:
                self.current_plasticity *= self.config.plasticity_boost_on_novelty
                self.current_plasticity = min(self.current_plasticity, 2.0)  # Cap
        
        # Track performance
        if loss is not None:
            self.performance_buffer.append(loss)
        
        # Update learning rates
        self._update_learning_rates()
        
        # Check newborn
        if self.step_count % self.config.newborn_min_steps == 0:
            self._maybe_newborn()
    
    def _update_learning_rates(self):
        """Update LR for each param group based on plasticity."""
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = group['initial_lr'] if 'initial_lr' in group else group['lr']
            new_lr = base_lr * self.current_plasticity * self.lr_multipliers[i]
            new_lr = max(self.config.min_lr, min(self.config.max_lr, new_lr))
            group['lr'] = new_lr
    
    def _maybe_newborn(self):
        """Reinitialize underperforming experts (newborn)."""
        # This would be called by ActiveLearner with model access
        pass
    
    def get_state(self) -> Dict:
        """Get controller state for checkpointing."""
        return {
            'step': self.step_count,
            'current_plasticity': self.current_plasticity,
            'lr_multipliers': self.lr_multipliers,
        }
    
    def load_state(self, state: Dict):
        """Load controller state."""
        self.step_count = state.get('step', 0)
        self.current_plasticity = state.get('current_plasticity', 1.0)
        self.lr_multipliers = state.get('lr_multipliers', [1.0] * len(self.optimizer.param_groups))


class ActiveLearner:
    """
    Active learning loop for continuous conversation training.
    Inspired by mini-AGI LiveLearner.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        plasticity_config: PlasticityConfig = None,
        optimizer: torch.optim.Optimizer = None,
        device: torch.device = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        self.config = plasticity_config or PlasticityConfig()
        self.optimizer = optimizer
        
        if optimizer is not None:
            # Store initial LRs
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
            self.plasticity = PlasticityController(self.config, optimizer)
        else:
            self.plasticity = None
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        self.max_history = 1000
        
        # Active learning buffer (for batch updates)
        self.learning_buffer: List[ConversationTurn] = []
        self.buffer_size = 32
        
        # Evaluation
        self.eval_dataset = None
        self.eval_interval = self.config.eval_interval
        
        logger.info("ActiveLearner initialized")
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to history."""
        self.conversation_history.append(turn)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Add to learning buffer
        self.learning_buffer.append(turn)
    
    def compute_novelty(self, tokens: torch.Tensor) -> float:
        """Compute novelty score based on embedding distance to history."""
        if len(self.conversation_history) < 10:
            return 1.0  # High novelty initially
        
        with torch.no_grad():
            # Get embeddings for current tokens
            current_emb = self.model.wte(tokens.to(self.device)).mean(dim=1)  # [B, D]
            
            # Compare with recent history
            similarities = []
            for turn in list(self.conversation_history)[-100:]:
                hist_emb = self.model.wte(turn.tokens.to(self.device)).mean(dim=1)
                sim = F.cosine_similarity(current_emb, hist_emb, dim=-1).mean().item()
                similarities.append(sim)
            
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            novelty = 1.0 - avg_sim
            
        return max(0.0, min(1.0, novelty))
    
    def learn_from_buffer(self) -> Dict[str, float]:
        """Perform learning step on buffered conversation turns."""
        if len(self.learning_buffer) < self.buffer_size:
            return {}
        
        if self.optimizer is None:
            logger.warning("No optimizer for active learning")
            return {}
        
        # Prepare batch from buffer
        batch_tokens = torch.cat([t.tokens for t in self.learning_buffer], dim=0)
        batch_levels = torch.cat([t.level for t in self.learning_buffer], dim=0)
        
        # Compute novelty for plasticity
        novelty = self.compute_novelty(batch_tokens)
        
        # Forward pass
        self.model.train()
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            logits, loss = self.model(batch_tokens, level=batch_levels)
            
            # Weight by reward if available
            rewards = torch.tensor([t.reward for t in self.learning_buffer], device=self.device)
            if rewards.abs().sum() > 0:
                loss = loss * (1.0 + rewards.mean())
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update plasticity
        if self.plasticity:
            self.plasticity.step(novelty_score=novelty, loss=loss.item())
        
        # Clear buffer
        self.learning_buffer.clear()
        
        return {
            'loss': loss.item(),
            'novelty': novelty,
            'plasticity': self.plasticity.current_plasticity if self.plasticity else 1.0,
            'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
        }
    
    def evaluate(self, eval_data: List[ConversationTurn]) -> Dict[str, float]:
        """Evaluate on held-out conversation turns."""
        if not eval_data:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for turn in eval_data[:self.config.eval_samples]:
                tokens = turn.tokens.to(self.device)
                level = turn.level.to(self.device)
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                    logits, loss = self.model(tokens, level=level)
                
                total_loss += loss.item() * tokens.numel()
                total_tokens += tokens.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
        }
    
    def save_checkpoint(self, path: str):
        """Save active learning state."""
        state = {
            'conversation_history': [
                {
                    'user_input': t.user_input,
                    'model_response': t.model_response,
                    'tokens': t.tokens,
                    'level': t.level,
                    'reward': t.reward,
                    'timestamp': t.timestamp,
                    'context': t.context,
                }
                for t in self.conversation_history
            ],
            'plasticity_state': self.plasticity.get_state() if self.plasticity else {},
            'step': len(self.conversation_history),
        }
        torch.save(state, path)
        logger.info(f"Active learning checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load active learning state."""
        state = torch.load(path, map_location='cpu')
        if self.plasticity:
            self.plasticity.load_state(state.get('plasticity_state', {}))
        
        # Restore conversation history
        self.conversation_history = []
        for item in state.get('conversation_history', []):
            turn = ConversationTurn(
                user_input=item['user_input'],
                model_response=item['model_response'],
                tokens=item['tokens'],
                level=item['level'],
                reward=item['reward'],
                timestamp=item['timestamp'],
                context=item.get('context', {}),
            )
            self.conversation_history.append(turn)
        
        logger.info(f"Active learning checkpoint loaded from {path}")


def create_active_learner(
    model: nn.Module,
    tokenizer,
    lr_trunk: float = 3e-4,
    lr_engram: float = 1e-4,
    lr_expert: float = 3e-3,
    device: torch.device = None,
) -> Tuple[ActiveLearner, torch.optim.Optimizer]:
    """Factory function to create ActiveLearner with optimizer."""
    
    # Create optimizer for active learning (smaller LR for stability)
    from training.train_moe_engram_2bit import get_optimizer_2bit
    optimizer = get_optimizer_2bit(
        model,
        lr_trunk=lr_trunk * 0.1,  # Lower LR for active learning
        lr_engram=lr_engram * 0.1,
        lr_expert=lr_expert * 0.1,
    )
    
    plasticity_config = PlasticityConfig(
        base_lr=lr_trunk * 0.1,
        min_lr=1e-6,
        max_lr=1e-3,
    )
    
    learner = ActiveLearner(
        model=model,
        tokenizer=tokenizer,
        plasticity_config=plasticity_config,
        optimizer=optimizer,
        device=device,
    )
    
    return learner, optimizer


if __name__ == "__main__":
    # Quick test
    import torch
    
    print("Testing ActiveLearner...")
    
    # Dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(1000, 128)
            self.lm_head = nn.Linear(128, 1000)
        
        def forward(self, x, level=None, targets=None):
            x = self.wte(x)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1)) if targets is not None else 0
            return logits, loss
    
    model = DummyModel()
    tokenizer = None
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create active learner
    config = PlasticityConfig(base_lr=1e-4)
    learner = ActiveLearner(model, tokenizer, config, optimizer)
    
    # Test novelty computation
    tokens = torch.randint(0, 1000, (2, 16))
    novelty = learner.compute_novelty(tokens)
    print(f"Novelty score: {novelty:.4f}")
    
    # Test learning step
    turn = ConversationTurn(
        user_input="Hello",
        model_response="Hi there!",
        tokens=tokens,
        level=torch.zeros_like(tokens),
    )
    learner.add_turn(turn)
    
    # Fill buffer
    for _ in range(32):
        learner.add_turn(turn)
    
    metrics = learner.learn_from_buffer()
    print(f"Learning metrics: {metrics}")
    
    print("ActiveLearner test passed!")
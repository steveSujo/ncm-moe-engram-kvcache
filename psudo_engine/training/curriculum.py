"""
Curriculum Learning for psudo-Engine.

Implements difficulty progression for psudo script training,
starting from simple constructs and gradually increasing complexity.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DifficultyLevel(Enum):
    """Curriculum difficulty levels."""
    BEGINNER = 1      # Basic syntax, simple functions
    INTERMEDIATE = 2  # Control flow, data structures
    ADVANCED = 3      # Complex patterns, file I/O
    EXPERT = 4        # Full programs, error handling


@dataclass
class CurriculumStage:
    """A single curriculum stage."""
    level: DifficultyLevel
    name: str
    description: str
    max_tokens: int
    allowed_keywords: list[str]
    allowed_constructs: list[str]
    weight: float = 1.0  # Sampling weight
    min_steps: int = 0   # Minimum steps before advancing
    max_steps: int = -1  # Maximum steps (-1 = unlimited)


class CurriculumScheduler:
    """
    Manages curriculum progression during training.

    Gradually increases difficulty based on training steps or validation metrics.
    """

    # Default curriculum stages
    DEFAULT_STAGES = [
        CurriculumStage(
            level=DifficultyLevel.BEGINNER,
            name="Basic Syntax",
            description="Simple variable assignments, basic functions, arithmetic",
            max_tokens=128,
            allowed_keywords=[
                "fn", "let", "return", "int", "float", "string", "bool",
                "+", "-", "*", "/", "=", "(", ")", "{", "}", ";",
            ],
            allowed_constructs=["variable_declaration", "function_def", "arithmetic"],
            weight=1.0,
            min_steps=500,
            max_steps=2000,
        ),
        CurriculumStage(
            level=DifficultyLevel.INTERMEDIATE,
            name="Control Flow",
            description="Conditionals, loops, basic data structures",
            max_tokens=256,
            allowed_keywords=[
                "if", "else", "while", "for", "in", "break", "continue",
                "true", "false", "<", ">", "<=", ">=", "==", "!=",
                "list", "dict", "range", "len",
            ],
            allowed_constructs=["if_else", "while_loop", "for_loop", "list_literal", "dict_literal"],
            weight=1.0,
            min_steps=2000,
            max_steps=5000,
        ),
        CurriculumStage(
            level=DifficultyLevel.ADVANCED,
            name="Data Structures & Functions",
            description="Nested structures, higher-order functions, file I/O",
            max_tokens=384,
            allowed_keywords=[
                "map", "filter", "zip", "enumerate", "print",
                "read_file", "write_file", "&&", "||", "!",
                "None", "is", "in", "not",
            ],
            allowed_constructs=["higher_order", "file_io", "nested_structures", "boolean_logic"],
            weight=1.0,
            min_steps=5000,
            max_steps=10000,
        ),
        CurriculumStage(
            level=DifficultyLevel.EXPERT,
            name="Full Programs",
            description="Complete programs with error handling, complex patterns",
            max_tokens=512,
            allowed_keywords=[],  # All keywords allowed
            allowed_constructs=[],  # All constructs allowed
            weight=1.0,
            min_steps=10000,
            max_steps=-1,
        ),
    ]

    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        schedule_type: str = "linear",  # linear, exponential, adaptive
        adaptive_threshold: float = 0.05,  # Val loss improvement threshold
    ):
        self.stages = stages or self.DEFAULT_STAGES
        self.schedule_type = schedule_type
        self.adaptive_threshold = adaptive_threshold

        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.total_steps = 0
        self.val_loss_history = []

        # Stage transitions
        self.stage_transitions = []

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def step(self, val_loss: float | None = None) -> bool:
        """
        Advance curriculum by one step.

        Args:
            val_loss: Optional validation loss for adaptive scheduling

        Returns:
            True if stage changed
        """
        self.total_steps += 1
        self.steps_in_stage += 1

        if val_loss is not None:
            self.val_loss_history.append(val_loss)

        # Check if should advance
        should_advance = self._should_advance()

        if should_advance and self.current_stage_idx < len(self.stages) - 1:
            self._advance_stage()
            return True

        return False

    def _should_advance(self) -> bool:
        """Determine if should advance to next stage."""
        stage = self.get_current_stage()

        # Check minimum steps
        if self.steps_in_stage < stage.min_steps:
            return False

        # Check maximum steps
        if stage.max_steps > 0 and self.steps_in_stage >= stage.max_steps:
            return True

        # Adaptive scheduling based on validation loss
        if self.schedule_type == "adaptive" and len(self.val_loss_history) >= 10:
            recent = self.val_loss_history[-10:]
            older = self.val_loss_history[-20:-10] if len(self.val_loss_history) >= 20 else recent

            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                improvement = (older_avg - recent_avg) / older_avg

                if improvement < self.adaptive_threshold:
                    return True

        return False

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        old_stage = self.get_current_stage()
        self.current_stage_idx += 1
        self.steps_in_stage = 0

        new_stage = self.get_current_stage()
        self.stage_transitions.append({
            'step': self.total_steps,
            'from_stage': old_stage.name,
            'to_stage': new_stage.name,
        })


    def get_sampling_weights(self) -> dict[str, float]:
        """Get sampling weights for current stage."""
        stage = self.get_current_stage()
        return {
            'weight': stage.weight,
            'max_tokens': stage.max_tokens,
            'allowed_keywords': stage.allowed_keywords,
            'allowed_constructs': stage.allowed_constructs,
        }

    def filter_sample(self, sample: dict[str, Any]) -> bool:
        """
        Filter sample based on current curriculum stage.

        Args:
            sample: Sample with 'code' field

        Returns:
            True if sample passes current stage filters
        """
        stage = self.get_current_stage()
        code = sample.get('code', '')

        # Check max tokens (rough estimate)
        if len(code) > stage.max_tokens * 4:  # ~4 chars per token
            return False

        # If no restrictions, allow all
        if not stage.allowed_keywords and not stage.allowed_constructs:
            return True

        # Check for disallowed keywords
        for keyword in stage.allowed_keywords:
            if keyword in code:
                # This is allowed, continue checking
                pass

        # In practice, would do more sophisticated AST-based filtering
        # For now, just use token length as proxy
        return True

    def get_progress(self) -> dict[str, Any]:
        """Get curriculum progress info."""
        stage = self.get_current_stage()
        return {
            'current_stage': stage.name,
            'stage_level': stage.level.value,
            'steps_in_stage': self.steps_in_stage,
            'total_steps': self.total_steps,
            'stage_progress': self.steps_in_stage / max(1, stage.min_steps),
            'transitions': self.stage_transitions,
        }

    def save_state(self, path: str):
        """Save curriculum state."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'steps_in_stage': self.steps_in_stage,
            'total_steps': self.total_steps,
            'val_loss_history': self.val_loss_history,
            'stage_transitions': self.stage_transitions,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load curriculum state."""
        with open(path) as f:
            state = json.load(f)

        self.current_stage_idx = state['current_stage_idx']
        self.steps_in_stage = state['steps_in_stage']
        self.total_steps = state['total_steps']
        self.val_loss_history = state['val_loss_history']
        self.stage_transitions = state['stage_transitions']


class SyntheticDataGenerator:
    """
    Generates synthetic psudo scripts for curriculum stages.
    """

    # Templates by difficulty
    TEMPLATES = {
        DifficultyLevel.BEGINNER: [
            'fn add(a: int, b: int) -> int {{\n    return a + b\n}}',
            'fn multiply(x: int, y: int) -> int {{\n    return x * y\n}}',
            'fn greet(name: string) -> string {{\n    return "Hello, " + name\n}}',
            'fn double(x: int) -> int {{\n    let y = x * 2\n    return y\n}}',
            'fn identity<T>(x: T) -> T {{\n    return x\n}}',
        ],
        DifficultyLevel.INTERMEDIATE: [
            'fn sum_range(start: int, end: int) -> int {{\n    let total = 0\n    for i in range(start, end) {{\n        total = total + i\n    }}\n    return total\n}}',
            'fn find_max(items: list<int>) -> int {{\n    let max_val = items[0]\n    for item in items {{\n        if item > max_val {{\n            max_val = item\n        }}\n    }}\n    return max_val\n}}',
            'fn factorial(n: int) -> int {{\n    if n <= 1 {{\n        return 1\n    }}\n    return n * factorial(n - 1)\n}}',
            'fn filter_even(numbers: list<int>) -> list<int> {{\n    let result = []\n    for n in numbers {{\n        if n % 2 == 0 {{\n            result.append(n)\n        }}\n    }}\n    return result\n}}',
        ],
        DifficultyLevel.ADVANCED: [
            'fn count_words(text: string) -> dict<string, int> {{\n    let words = text.split(" ")\n    let counts = dict()\n    for word in words {{\n        if word in counts {{\n            counts[word] = counts[word] + 1\n        }} else {{\n            counts[word] = 1\n        }}\n    }}\n    return counts\n}}',
            'fn read_and_process(path: string) -> string {{\n    let content = read_file(path)\n    let lines = content.split("\\n")\n    let processed = lines.map(|line| line.upper())\n    return processed.join("\\n")\n}}',
            'fn write_report(path: string, data: dict<string, int>) -> () {{\n    let lines = []\n    for key, value in data {{\n        lines.append(key + ": " + str(value))\n    }}\n    write_file(path, lines.join("\\n"))\n}}',
        ],
        DifficultyLevel.EXPERT: [
            'fn process_files(input_dir: string, output_dir: string) -> () {{\n    let files = list_files(input_dir)\n    for file in files {{\n        let content = read_file(input_dir + "/" + file)\n        let result = transform(content)\n        write_file(output_dir + "/out_" + file, result)\n    }}\n}}',
            'fn server_loop(port: int) -> () {{\n    let server = start_server(port)\n    while true {{\n        let request = server.accept()\n        let response = handle_request(request)\n        server.respond(response)\n    }}\n}}',
            'fn parse_json(json_str: string) -> dict {{\n    // Complex parsing logic\n    return parse(json_str)\n}}',
        ],
    }

    def __init__(self):
        pass

    def generate_for_stage(
        self,
        stage: CurriculumStage,
        num_samples: int
    ) -> list[dict[str, Any]]:
        """Generate synthetic samples for a curriculum stage."""
        templates = self.TEMPLATES.get(stage.level, self.TEMPLATES[DifficultyLevel.BEGINNER])

        samples = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            samples.append({
                'code': template,
                'level': stage.level.name,
                'stage': stage.name,
                'id': f'synthetic_{stage.level.name}_{i}',
            })

        return samples

    def generate_progressive(
        self,
        stages: list[CurriculumStage],
        samples_per_stage: int
    ) -> list[dict[str, Any]]:
        """Generate samples across all stages."""
        all_samples = []
        for stage in stages:
            samples = self.generate_for_stage(stage, samples_per_stage)
            all_samples.extend(samples)
        return all_samples


def create_curriculum_scheduler(config: dict[str, Any] | None = None) -> CurriculumScheduler:
    """Create CurriculumScheduler from config."""
    config = config or {}

    schedule_type = config.get('schedule_type', 'linear')
    adaptive_threshold = config.get('adaptive_threshold', 0.05)

    return CurriculumScheduler(
        schedule_type=schedule_type,
        adaptive_threshold=adaptive_threshold
    )


if __name__ == "__main__":
    # Quick test

    scheduler = CurriculumScheduler()

    # Simulate training steps
    for i in range(25):
        # Simulate validation loss decreasing
        val_loss = 5.0 - i * 0.15
        changed = scheduler.step(val_loss)

        if changed:
            progress = scheduler.get_progress()

    final_progress = scheduler.get_progress()

    # Test data generator
    generator = SyntheticDataGenerator()
    stages = scheduler.stages
    samples = generator.generate_progressive(stages, 2)


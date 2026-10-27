"""
Schema Version Check for psudo-Engine.

Ensures training and transpiler use compatible psudo-spec versions.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SchemaInfo:
    """Schema version information."""
    version: str
    spec_path: str
    loaded_at: str
    checksum: str


class SchemaValidator:
    """
    Validates schema version compatibility.

    Used by both training pipeline and transpiler to ensure
    they're using the same psudo-spec version.
    """

    def __init__(self, spec_path: str | None = None):
        if spec_path is None:
            # Default to psudo_engine/data/psudo-spec.json
            spec_path = Path(__file__).parent.parent / "data" / "psudo-spec.json"

        self.spec_path = Path(spec_path)
        self._schema: dict[str, Any] | None = None
        self._schema_info: SchemaInfo | None = None

    def load_schema(self) -> dict[str, Any]:
        """Load and cache schema."""
        if self._schema is None:
            with open(self.spec_path) as f:
                self._schema = json.load(f)
        return self._schema

    def get_version(self) -> str:
        """Get schema version."""
        schema = self.load_schema()
        return schema.get("version", "0.0.0")

    def get_schema_info(self) -> SchemaInfo:
        """Get schema info with checksum."""
        if self._schema_info is None:
            import datetime
            import hashlib

            content = self.spec_path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()[:16]

            self._schema_info = SchemaInfo(
                version=self.get_version(),
                spec_path=str(self.spec_path),
                loaded_at=datetime.datetime.utcnow().isoformat() + "Z",
                checksum=checksum
            )
        return self._schema_info

    def validate_version(self, expected_version: str) -> bool:
        """
        Validate that loaded schema matches expected version.

        Args:
            expected_version: Expected schema version

        Returns:
            True if versions match
        """
        actual = self.get_version()

        return actual == expected_version

    def validate_training_config(self, config: dict[str, Any]) -> bool:
        """Validate training config has correct schema version."""
        schema_version = config.get("schema_version", "0.0.0")
        return self.validate_version(schema_version)

    def validate_transpiler_config(self, config: dict[str, Any]) -> bool:
        """Validate transpiler config has correct schema version."""
        # Transpiler config might have version in different location
        spec_version = config.get("transpiler", {}).get("spec_version", "0.0.0")
        return self.validate_version(spec_version)

    def check_compatibility(
        self,
        training_config: dict[str, Any],
        transpiler_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Check compatibility between training and transpiler configs.

        Returns:
            Dict with compatibility info
        """
        training_version = training_config.get("schema_version", "0.0.0")
        transpiler_version = transpiler_config.get("transpiler", {}).get("spec_version", "0.0.0")
        schema_version = self.get_version()

        compatible = (training_version == schema_version and
                     transpiler_version == schema_version)

        return {
            "compatible": compatible,
            "schema_version": schema_version,
            "training_version": training_version,
            "transpiler_version": transpiler_version,
            "mismatches": [
                k for k, v in {
                    "training": training_version,
                    "transpiler": transpiler_version,
                    "schema": schema_version
                }.items() if v != schema_version
            ]
        }

    def print_compatibility_report(
        self,
        training_config: dict[str, Any] | None = None,
        transpiler_config: dict[str, Any] | None = None
    ):
        """Print compatibility report."""
        self.get_schema_info()


        if training_config:
            training_config.get("schema_version", "unknown")

        if transpiler_config:
            transpiler_config.get("transpiler", {}).get("spec_version", "unknown")

        if training_config and transpiler_config:
            result = self.check_compatibility(training_config, transpiler_config)
            if result["mismatches"]:
                pass


def create_validator(spec_path: str | None = None) -> SchemaValidator:
    """Create SchemaValidator instance."""
    return SchemaValidator(spec_path)


# Expected schema version (should match psudo-spec.json)
PSUDO_SPEC_VERSION = "1.0.0"


def check_schema_version(config: dict[str, Any], config_type: str = "training") -> bool:
    """
    Quick schema version check.

    Args:
        config: Configuration dict
        config_type: "training" or "transpiler"

    Returns:
        True if version matches
    """
    validator = SchemaValidator()

    if config_type == "training":
        return validator.validate_training_config(config)
    elif config_type == "transpiler":
        return validator.validate_transpiler_config(config)
    else:
        return validator.validate_version(config.get("schema_version", "0.0.0"))


if __name__ == "__main__":
    # Quick test

    validator = SchemaValidator()
    validator.print_compatibility_report()

    # Test with mock configs
    training_cfg = {"schema_version": "1.0.0"}
    transpiler_cfg = {"transpiler": {"spec_version": "1.0.0"}}

    result = validator.check_compatibility(training_cfg, transpiler_cfg)


#!/usr/bin/env python3
"""
AI Model Reproducibility Testing Script

Tests the same prompt across different AI models and logs results for comparison.
Results are saved to a JSON file with timestamps for tracking reproducibility across time.
"""

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for AI model providers"""

    @abstractmethod
    def query(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        config: Dict[str, Any]
            parser = argparse.ArgumentParser(
                description="Test AI models with the same prompt and compare results"
            )

            parser.add_argument(
                "--models",
                default="config/models.json",
                help="Path to models configuration file (default: config/models.json)"
            )

            parser.add_argument(
                "--system-prompt",
                default="prompts/system_prompt.txt",
                help="Path to system prompt file (default: prompts/system_prompt.txt)"
            )

            parser.add_argument(
                "--user-prompt",
                default="prompts/user_prompt.txt",
                help="Path to user prompt file (default: prompts/user_prompt.txt)"
            )

            parser.add_argument(
                "--output",
                help="Path to output JSON file (auto-generated if not provided)"
            )

            parser.add_argument(
                "--openai-api-key",
                help="OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)"
            )

            args = parser.parse_args()

            try:
                tester = ModelTester(
                    models_config_path=args.models,
                    system_prompt_path=args.system_prompt,
                    user_prompt_path=args.user_prompt,
                    output_path=args.output,
                    openai_api_key=args.openai_api_key
                )
                tester.run_tests()

            except Exception as e:
                logger.error(f"Fatal error: {str(e)}")
                sys.exit(1)
                max_tokens=config.get("max_tokens", 2000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying {model_name}: {str(e)}")
            raise


class ModelProviderFactory:
    """Factory for creating model providers"""

    _providers: Dict[str, type] = {
        "openai": OpenAIProvider
    }
    _api_keys: Dict[str, Optional[str]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new provider"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def set_api_key(cls, provider: str, api_key: str) -> None:
        """Set API key for a provider"""
        cls._api_keys[provider.lower()] = api_key

    @classmethod
    def create_provider(cls, provider_name: str) -> ModelProvider:
        """Create a provider instance"""
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(cls._providers.keys())}"
            )
        api_key = cls._api_keys.get(provider_name.lower())
        return provider_class(api_key=api_key) if provider_name.lower() == "openai" else provider_class()


class ModelTester:
    """Main class for testing models"""

    def __init__(
        self,
        models_config_path: str,
        system_prompt_path: str,
        user_prompt_path: str,
        output_path: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the tester.
        
        Args:
            models_config_path: Path to models.json configuration
            system_prompt_path: Path to system prompt file
            user_prompt_path: Path to user prompt file
            output_path: Path to output JSON file (auto-generated if not provided)
            openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        """
        self.models_config_path = Path(models_config_path)
        self.system_prompt_path = Path(system_prompt_path)
        self.user_prompt_path = Path(user_prompt_path)
        
        # Set API key if provided
        if openai_api_key:
            ModelProviderFactory.set_api_key("openai", openai_api_key)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/results_{timestamp}.json"
        self.output_path = Path(output_path)
        
        # Load configuration
        self.models_config = self._load_models_config()
        self.system_prompt = self._load_file(self.system_prompt_path)
        self.user_prompt = self._load_file(self.user_prompt_path)

    @staticmethod
    def _load_file(path: Path) -> str:
        """Load content from a file"""
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise

    def _load_models_config(self) -> List[Dict[str, Any]]:
        """Load models configuration from JSON"""
        try:
            with open(self.models_config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            return config.get("models", [])
        except FileNotFoundError:
            logger.error(f"Models config file not found: {self.models_config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in models config: {self.models_config_path}")
            raise

    def run_tests(self) -> None:
        """Run tests across all configured models"""
        logger.info(f"Starting tests with {len(self.models_config)} models")
        logger.info(f"System prompt: {self.system_prompt[:100]}...")
        logger.info(f"User prompt: {self.user_prompt}")
        
        results = []
        
        for model_config in self.models_config:
            model_name = model_config.get("name")
            provider_name = model_config.get("provider")
            config = model_config.get("config", {})
            
            logger.info(f"Testing model: {model_name} (provider: {provider_name})")
            
            try:
                # Create provider and query
                provider = ModelProviderFactory.create_provider(provider_name)
                response = provider.query(
                    model_name=model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt,
                    config=config
                )
                
                # Record result
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "provider": provider_name,
                    "system_prompt_hash": hash(self.system_prompt),
                    "user_prompt_hash": hash(self.user_prompt),
                    "temperature": config.get("temperature", "N/A"),
                    "max_tokens": config.get("max_tokens", "N/A"),
                    "response": response,
                    "response_length": len(response),
                    "status": "success"
                }
                
                logger.info(f"✓ {model_name}: Success ({len(response)} chars)")
                results.append(result)
                
            except Exception as e:
                # Record error
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "provider": provider_name,
                    "system_prompt_hash": hash(self.system_prompt),
                    "user_prompt_hash": hash(self.user_prompt),
                    "temperature": config.get("temperature", "N/A"),
                    "max_tokens": config.get("max_tokens", "N/A"),
                    "response": f"ERROR: {str(e)}",
                    "response_length": 0,
                    "status": "error"
                }
                
                logger.error(f"✗ {model_name}: {str(e)}")
                results.append(result)
        
        # Save results
        self._save_results(results)
        logger.info(f"JSON results appended to: {self.output_path}")

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Append results to cumulative JSON file"""
        if not results:
            logger.warning("No results to save")
            return
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON storage
        json_results = []
        for result in results:
            json_result = {
                "timestamp": result["timestamp"],
                "model": result["model_name"],
                "provider": result["provider"],
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "response": result["response"],
                "temperature": result["temperature"],
                "max_tokens": result["max_tokens"],
                "response_length": result["response_length"],
                "status": result["status"]
            }
            json_results.append(json_result)
        
        # Load existing results if file exists
        all_results = []
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                    if not isinstance(all_results, list):
                        all_results = [all_results]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing JSON file {self.output_path}: {str(e)}")
                all_results = []
        
        # Append new results
        all_results.extend(json_results)
        
        # Write back to JSON file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test AI models with the same prompt and compare results"
    )
    
    parser.add_argument(
        "--models",
        default="config/models.json",
        help="Path to models configuration file (default: config/models.json)"
    )
    
    parser.add_argument(
        "--system-prompt",
        default="prompts/system_prompt.txt",
        help="Path to system prompt file (default: prompts/system_prompt.txt)"
    )
    
    parser.add_argument(
        "--output",
        help="Path to output CSV file (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    parser.add_argument(
        "--output",
    try:
        tester = ModelTester(
    parser.add_argument(
        "--output",
        help="Path to output JSON file (auto-generated if not provided)"
    )       output_path=args.output,
            openai_api_key=args.openai_api_key
        )
        tester.run_tests()path=args.models,
            system_prompt_path=args.system_prompt,
            user_prompt_path=args.user_prompt,
            output_path=args.output
        )
        tester.run_tests()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

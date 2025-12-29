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

# Configure logging with both console and daily file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Daily log file with date
log_filename = log_dir / f"test_run_{datetime.now().strftime('%Y%m%d')}.log"

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# File handler
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""

    @abstractmethod
    def query(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        config: Dict[str, Any],
    ) -> str:
        """Query the model and return the response."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise

    def query(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        config: Dict[str, Any],
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _token_value() -> Optional[int]:
            return (
                config.get("max_completion_tokens")
                or config.get("max_output_tokens")
                or config.get("max_tokens")
            )

        def chat_completion(use_max_completion_tokens: bool = False) -> str:
            params: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }

            # Only add temperature if specified in config
            if "temperature" in config:
                params["temperature"] = config["temperature"]

            token_val = _token_value()
            if token_val:
                if use_max_completion_tokens:
                    params["max_completion_tokens"] = token_val
                else:
                    params["max_tokens"] = token_val

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

        def completions(use_max_completion_tokens: bool = False) -> str:
            # Fallback for models that require the /v1/completions endpoint
            params: Dict[str, Any] = {
                "model": model_name,
                "prompt": f"System: {system_prompt}\nUser: {user_prompt}",
            }

            # Only add temperature if specified in config
            if "temperature" in config:
                params["temperature"] = config["temperature"]

            token_val = _token_value()
            if token_val:
                if use_max_completion_tokens:
                    params["max_completion_tokens"] = token_val
                else:
                    params["max_tokens"] = token_val

            response = self.client.completions.create(**params)
            return response.choices[0].text

        def responses_api() -> str:
            # Fallback for models supported only by the /v1/responses endpoint
            params: Dict[str, Any] = {
                "model": model_name,
                "input": messages,
            }

            # Only add temperature if specified in config
            if "temperature" in config:
                params["temperature"] = config["temperature"]

            token_val = _token_value()
            if token_val:
                params["max_output_tokens"] = token_val

            response = self.client.responses.create(**params)
            return self._extract_response_text(response)

        last_error: Optional[Exception] = None

        # 1) Try chat completions with legacy max_tokens
        try:
            return chat_completion(use_max_completion_tokens=False)
        except Exception as e:
            last_error = e
            message = str(e).lower()

            # 2) Retry chat with the new max_completion_tokens parameter if hinted
            if "max_completion_tokens" in message or "unsupported parameter: 'max_tokens'" in message:
                try:
                    return chat_completion(use_max_completion_tokens=True)
                except Exception as e2:
                    last_error = e2
                    message = str(e2).lower()

            # 3) If the model is not a chat model, try /v1/completions
            if "not a chat model" in message or "v1/completions" in message:
                try:
                    return completions(use_max_completion_tokens="max_completion_tokens" in message)
                except Exception as e3:
                    last_error = e3
                    message = str(e3).lower()

            # 4) If the API suggests /v1/responses, try that endpoint
            if "v1/responses" in message:
                try:
                    return responses_api()
                except Exception as e4:
                    last_error = e4

        logger.error(f"Error querying {model_name}: {str(last_error) if last_error else 'Unknown error'}")
        raise last_error if last_error else RuntimeError("Unknown error while querying model")

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        # The Responses API can shape-shift; try common access patterns before falling back to str().
        for attr in ("output_text", "text", "content", "message"):
            val = getattr(response, attr, None)
            if isinstance(val, str):
                return val

        if hasattr(response, "output"):
            output = getattr(response, "output")
            try:
                # Typical shape: response.output[0].content[0].text
                return output[0].content[0].text  # type: ignore[index]
            except Exception:
                pass

        return str(response)


class ModelProviderFactory:
    """Factory for creating model providers."""

    _providers: Dict[str, type] = {"openai": OpenAIProvider}
    _api_keys: Dict[str, Optional[str]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        cls._providers[name.lower()] = provider_class

    @classmethod
    def set_api_key(cls, provider: str, api_key: str) -> None:
        cls._api_keys[provider.lower()] = api_key

    @classmethod
    def create_provider(cls, provider_name: str) -> ModelProvider:
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(cls._providers.keys())}"
            )
        api_key = cls._api_keys.get(provider_name.lower())
        if provider_name.lower() == "openai":
            return provider_class(api_key=api_key)
        return provider_class()


class ModelTester:
    """Main class for testing models."""

    def __init__(
        self,
        models_config_path: str,
        system_prompt_path: str,
        user_prompt_path: str,
        output_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.models_config_path = Path(models_config_path)
        self.system_prompt_path = Path(system_prompt_path)
        self.user_prompt_path = Path(user_prompt_path)

        if openai_api_key:
            ModelProviderFactory.set_api_key("openai", openai_api_key)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/results_{timestamp}.json"
        self.output_path = Path(output_path)

        self.models_config = self._load_models_config()
        self.system_prompt = self._load_file(self.system_prompt_path)
        self.user_prompt = self._load_file(self.user_prompt_path)

    @staticmethod
    def _load_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise

    def _load_models_config(self) -> List[Dict[str, Any]]:
        try:
            with open(self.models_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config.get("models", [])
        except FileNotFoundError:
            logger.error(f"Models config file not found: {self.models_config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in models config: {self.models_config_path}")
            raise

    @staticmethod
    def _token_setting(config: Dict[str, Any]) -> Any:
        return (
            config.get("max_completion_tokens")
            or config.get("max_output_tokens")
            or config.get("max_tokens")
            or "N/A"
        )

    def run_tests(self) -> None:
        logger.info(f"Starting tests with {len(self.models_config)} models")
        logger.info(f"System prompt: {self.system_prompt[:100]}...")
        logger.info(f"User prompt: {self.user_prompt}")

        results: List[Dict[str, Any]] = []

        for model_config in self.models_config:
            model_name = model_config.get("name")
            provider_name = model_config.get("provider")
            config = model_config.get("config", {})
            token_setting = self._token_setting(config)
            
            # Skip disabled models
            if model_config.get("disabled", False):
                logger.info(f"Skipping disabled model: {model_name} (provider: {provider_name})")
                if model_config.get("note"):
                    logger.info(f"  Note: {model_config['note']}")
                continue

            logger.info(f"Testing model: {model_name} (provider: {provider_name})")

            try:
                provider = ModelProviderFactory.create_provider(provider_name)
                response = provider.query(
                    model_name=model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=self.user_prompt,
                    config=config,
                )

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "provider": provider_name,
                    "system_prompt_hash": hash(self.system_prompt),
                    "user_prompt_hash": hash(self.user_prompt),
                    "temperature": config.get("temperature", "N/A"),
                    "max_tokens": token_setting,
                    "response": response,
                    "response_length": len(response),
                    "status": "success",
                }

                logger.info(f"✓ {model_name}: Success ({len(response)} chars)")
                results.append(result)

            except Exception as e:
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "provider": provider_name,
                    "system_prompt_hash": hash(self.system_prompt),
                    "user_prompt_hash": hash(self.user_prompt),
                    "temperature": config.get("temperature", "N/A"),
                    "max_tokens": token_setting,
                    "response": f"ERROR: {str(e)}",
                    "response_length": 0,
                    "status": "error",
                }

                logger.error(f"✗ {model_name}: {str(e)}")
                results.append(result)

        self._save_results(results)
        logger.info(f"JSON results appended to: {self.output_path}")

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            logger.warning("No results to save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        json_results: List[Dict[str, Any]] = []
        for result in results:
            json_results.append(
                {
                    "timestamp": result["timestamp"],
                    "model": result["model_name"],
                    "provider": result["provider"],
                    "system_prompt": self.system_prompt,
                    "user_prompt": self.user_prompt,
                    "response": result["response"],
                    "temperature": result["temperature"],
                    "max_tokens": result["max_tokens"],
                    "response_length": result["response_length"],
                    "status": result["status"],
                }
            )

        all_results: List[Dict[str, Any]] = []
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        all_results = loaded
                    else:
                        all_results = [loaded]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing JSON file {self.output_path}: {str(e)}")
                all_results = []

        all_results.extend(json_results)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Test AI models with the same prompt and compare results"
    )

    parser.add_argument(
        "--models",
        default="config/models.json",
        help="Path to models configuration file (default: config/models.json)",
    )

    parser.add_argument(
        "--system-prompt",
        default="prompts/system_prompt.txt",
        help="Path to system prompt file (default: prompts/system_prompt.txt)",
    )

    parser.add_argument(
        "--user-prompt",
        default="prompts/user_prompt.txt",
        help="Path to user prompt file (default: prompts/user_prompt.txt)",
    )

    parser.add_argument(
        "--output",
        help="Path to output JSON file (auto-generated if not provided)",
    )

    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)",
    )

    args = parser.parse_args()

    try:
        tester = ModelTester(
            models_config_path=args.models,
            system_prompt_path=args.system_prompt,
            user_prompt_path=args.user_prompt,
            output_path=args.output,
            openai_api_key=args.openai_api_key,
        )
        tester.run_tests()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

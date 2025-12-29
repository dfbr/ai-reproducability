# AI Reproducibility Testing

Test the same prompt across different AI models and track results over time to evaluate reproducibility.

## Features

- **Multi-model testing**: Query different AI models with the same prompt
- **Extensible architecture**: Easy to add new model providers (OpenAI, Anthropic, etc.)
- **External configuration**: Prompts and models defined in separate files
- **Result tracking**: JSON file that accumulates results across multiple test runs
- **Error handling**: Graceful error handling with detailed logging

## Project Structure

```
ai-reproducability/
├── test_models.py              # Main testing script
├── config/
│   └── models.json             # Model configuration (provider, name, parameters)
├── prompts/
│   ├── system_prompt.txt       # System prompt/instructions
│   └── user_prompt.txt         # User prompt/question
├── results/                    # Output JSON files (auto-generated with timestamps)
└── requirements.txt            # Python dependencies
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Provide your OpenAI API key** (choose one method):
   - **Via command-line option** (recommended):
     ```bash
     python test_models.py --openai-api-key "your-api-key-here"
     ```
   - **Via environment variable**:
     ```bash
     $env:OPENAI_API_KEY = "your-api-key-here"  # PowerShell
     # or
     export OPENAI_API_KEY="your-api-key-here"  # Bash
     ```

## Usage

### Basic Usage

```bash
python test_models.py
```

This uses default paths:
- Models: `config/models.json`
- System prompt: `prompts/system_prompt.txt`
- User prompt: `prompts/user_prompt.txt`
- Output: `results/results_YYYYMMDD_HHMMSS.json` (auto-generated)

### Custom Paths

```bash
python test_models.py --models custom_models.json --system-prompt my_system.txt --user-prompt my_user.txt --output my_results.json --openai-api-key "your-api-key"
```

### Command-line Options

- `--models PATH`: Path to models configuration JSON (default: `config/models.json`)
- `--system-prompt PATH`: Path to system prompt file (default: `prompts/system_prompt.txt`)
- `--user-prompt PATH`: Path to user prompt file (default: `prompts/user_prompt.txt`)
- `--output PATH`: Output JSON file path (auto-generated if not specified)
- `--openai-api-key KEY`: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

## Configuration Files

### models.json

Define models and their providers in `config/models.json`:

```json
{
  "models": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "config": {
        "temperature": 0.7,
        "max_tokens": 2000
      }
    },
    {
      "name": "gpt-3.5-turbo",
      "provider": "openai",
      "config": {
        "temperature": 0.7,
        "max_tokens": 2000
      }
    }
  ]
}
```

**Note on temperature settings:** Different models use different temperature values in the configuration:
- Most models use `temperature: 0.7` (a common default balancing creativity and focus)
- Some models (GPT-5 series variants and o-series models) use `temperature: 1` (their required/recommended default)

This is intentional - each model is tested with its optimal temperature setting rather than forcing uniform values across all models. Since the goal is to observe what models produce rather than strict experimental control, using each model's natural operating temperature provides more representative results.

**Note on enabled property:**

Each model in `models.json` now includes an `enabled` property (boolean). Only models with `"enabled": true` will be tested. To temporarily exclude a model from test runs (without deleting its configuration), set `"enabled": false` for that model. This is useful for disabling expensive or experimental models while keeping their settings for future use.

Example:

```json
{
  "name": "o3",
  "provider": "openai",
  "enabled": false,
  "config": { "temperature": 1, "max_completion_tokens": 100000 },
  "note": "Reasoning model - disabled due to high cost"
}
```

### system_prompt.txt & user_prompt.txt

Plain text files containing the prompts:

```
You are a helpful AI assistant. Provide clear and accurate responses.
```

## Output Format

Results are saved to a JSON file that accumulates across runs. Each run appends new entries to the file, creating a historical record for comparison.

### JSON File (results_YYYYMMDD_HHMMSS.json)

Each entry contains:

- `timestamp` - When the query was made (ISO 8601 format)
- `model` - Model name (e.g., "gpt-4")
- `provider` - Provider name (e.g., "openai")
- `system_prompt` - Full system prompt text
- `user_prompt` - Full user prompt text
- `response` - Complete model response
- `temperature` - Temperature parameter used
- `max_tokens` - Max tokens parameter used
- `response_length` - Character count of response
- `status` - "success" or "error"

**Key advantages:**
- Compare responses across different models with identical prompts
- Track how a model's answers change (or stay consistent) over time
- Full historical record for reproducibility analysis
- Easy to programmatically compare and analyze responses

## Extensibility

### Adding a New Provider

To add support for a new model provider (e.g., Anthropic, Google):

1. Create a new provider class inheriting from `ModelProvider`:

```python
class AnthropicProvider(ModelProvider):
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()

    def query(self, model_name, system_prompt, user_prompt, config):
        response = self.client.messages.create(
            model=model_name,
            max_tokens=config.get("max_tokens", 2000),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
```

2. Register the provider in your code:

```python
ModelProviderFactory.register_provider("anthropic", AnthropicProvider)
```

3. Update `models.json` to include models from the new provider:

```json
{
  "name": "claude-3-opus",
  "provider": "anthropic",
  "config": {"max_tokens": 2000}
}
```

## Example Workflow

1. **Edit prompts**:
   - Modify `prompts/user_prompt.txt` with your question
   - Modify `prompts/system_prompt.txt` with your instructions

2. **Configure models**:
   - Edit `config/models.json` to specify which models to test

3. **Run tests**:
   ```bash
   python test_models.py
   ```

4. **Review results**:
   - Open the generated JSON in `results/results_YYYYMMDD_HHMMSS.json`
   - Compare responses across models and over time

## Notes

- Results are appended to the JSON file on each run, creating a complete historical record
- Each run with the same prompts allows tracking reproducibility over time
- Errors in individual model queries don't stop the entire test run
- Full response text is preserved for detailed analysis
- All logging is printed to console for real-time feedback

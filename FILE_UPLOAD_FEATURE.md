# File Upload API Feature

This branch adds support for uploading large data files to AI models using OpenAI's Assistants API.

## What's New

### File Upload Support
- Upload data files (CSV, JSON, TXT, etc.) of any size (multi-MB files supported)
- Uses OpenAI's Assistants API with file_search capability
- Automatic file upload, processing, and cleanup
- File upload results are cached per session to avoid redundant uploads

### New Command-Line Option

```bash
python test_models.py --data-file path/to/your/data.csv
```

### How It Works

1. **Without data file**: Uses the standard Chat Completions API (same as before)
2. **With data file**: Automatically switches to Assistants API:
   - Uploads the file to OpenAI
   - Creates a temporary assistant with file_search tool
   - Processes the query with the uploaded file
   - Cleans up assistant and thread after completion
   - Caches file ID for reuse during the same run

### API Architecture

The script now supports two query modes:

#### Standard Mode (No File)
```
User Prompt → Chat Completions API → Response
```

#### File Upload Mode
```
Data File → Upload → Assistant API → File Search → Response
     ↓
  Cached for reuse
```

### Example Usage

```bash
# Analyze a large CSV file
python test_models.py \
  --user-prompt prompts/analyze_data.txt \
  --data-file data/sales_report.csv

# Analyze a JSON configuration
python test_models.py \
  --user-prompt prompts/what_insights.txt \
  --data-file data/metrics.json
```

### Output Format

Results now include a `data_file` field when files are uploaded:

```json
{
  "timestamp": "2026-01-06T20:30:00",
  "model": "gpt-4o",
  "provider": "openai",
  "data_file": "data/sales_report.csv",
  "response": "Based on the provided data...",
  ...
}
```

### Logging

Enhanced logging shows file upload progress:

```
INFO - Testing model: gpt-4o (provider: openai)
INFO -   Using data file: sales_report.csv (2.34 MB)
INFO - Uploading file: sales_report.csv (2400.56 KB)
INFO - File uploaded successfully with ID: file-abc123...
INFO - Creating assistant with model gpt-4o
INFO - Running assistant on thread thread_xyz...
INFO - ✓ Assistant response received (1523 chars)
```

### Provider Support

Currently supported:
- ✅ OpenAI (via Assistants API)

Future providers can implement the `supports_file_upload()` method to enable this feature.

### Technical Details

**Key Changes:**
- Added `data_file_path` parameter throughout the call chain
- New `_upload_file()` method with caching
- New `_query_with_file()` method for Assistants API
- Automatic cleanup of temporary assistants and threads
- Provider capability detection via `supports_file_upload()`

**File Size Limits:**
- OpenAI Assistants API: Up to 512 MB per file
- Recommended: < 100 MB for faster processing

**Supported File Types:**
- Text: .txt, .md, .csv, .json, .xml, etc.
- Documents: .pdf, .docx (text extraction)
- Code: .py, .js, .java, etc.

### Known Limitations

1. Assistants API is slower than Chat Completions (creates temp resources)
2. Files are uploaded once per run, not persisted across runs
3. Only OpenAI provider supports file uploads currently
4. Additional costs may apply for file storage and retrieval

### Migration Notes

For existing scripts:
- **No breaking changes** - file upload is optional
- All existing functionality remains unchanged
- Simply add `--data-file` to enable new feature

### Future Enhancements

Potential improvements:
- Persistent file storage across runs
- Support for multiple files
- Support for other providers (Anthropic, Google, etc.)
- Image/PDF vision support
- Streaming responses from Assistants API

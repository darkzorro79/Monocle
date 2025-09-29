# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Ollama Integration** (`monocle-ollama` command)
  - Full support for Ollama API backend
  - Works with RTX 5090 and other modern GPUs (Blackwell architecture)
  - Supports local and remote Ollama servers
  - 10-30x faster inference compared to CPU mode

- **Multi-language Support**
  - `--language` parameter for English/Russian output
  - Localized explanations from LLM

- **Model Selection**
  - `--model` parameter to choose any HuggingFace or Ollama model
  - Default: `mistralai/Mistral-7B-Instruct-v0.2` (HuggingFace)
  - Default Ollama: `qwen2.5-coder:7b`

- **Improved Ghidra Integration**
  - Better error messages when Ghidra not found
  - Clear prompt for `analyzeHeadless.bat` path
  - Pre-check before starting analysis

- **RTX 5090 Support**
  - Automatic GPU compatibility detection
  - Graceful fallback to CPU when GPU incompatible with PyTorch
  - Full GPU support via Ollama backend

### Fixed
- **Robust LLM Response Parsing**
  - Handles multiple response formats (`3:`, `Score: 3`, `3`, etc.)
  - Prevents crashes on unexpected output
  - Score validation and clamping (0-10)

- **HuggingFace Authentication**
  - Support for `--token` parameter
  - Support for `HF_TOKEN` environment variable
  - Clear error messages for missing/invalid tokens

- **Setup Issues**
  - Fixed package discovery with `find_packages()`
  - Excluded `build/` and `coverage/` from package

### Changed
- Improved model loading with `device_map="auto"` for better GPU utilization
- Enhanced table output with color-coded scores
- Better error handling throughout the codebase

### Documentation
- Added comprehensive Russian documentation:
  - `ИНСТРУКЦИЯ_НАСТРОЙКИ.md` - Setup instructions
  - `КРАТКАЯ_ИНСТРУКЦИЯ.md` - Quick start guide
  - `ПРОБЛЕМА_RTX_5090.md` - RTX 5090 compatibility info
  - `МОДЕЛИ_И_ИСПОЛЬЗОВАНИЕ.md` - Models guide
  - `OLLAMA_GUIDE.md` - Ollama integration guide

## [0.1] - Original Release

### Features
- Binary analysis using Ghidra decompilation
- Natural language search using Mistral-7B-Instruct model
- 4-bit quantization for efficient GPU usage
- Score-based ranking (0-10) of functions
- Live updating results table
- Support for analyzing authentication code, encryption, vulnerabilities, etc.

# CSV Data Analyst - Setup and Configuration Guide

This guide provides comprehensive instructions for setting up and configuring the CSV Data Analyst application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Environment Variables](#environment-variables)
5. [Validation and Health Checks](#validation-and-health-checks)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 1GB free space

### Required Accounts and Keys

- **Google Cloud Account**: Required for Gemini API access
- **Google API Key**: Gemini 1.5 Flash API key with appropriate permissions

## Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd csv-data-analyst

# Or download and extract the project files
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Set Up Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual configuration values
# (See Environment Variables section below)
```

## Configuration

### Google API Key Setup

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Gemini API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Generative Language API"
   - Click "Enable"

3. **Create API Key**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key
   - (Optional) Restrict the key to specific APIs for security

4. **Add to Configuration**:
   - Open your `.env` file
   - Replace `your_google_api_key_here` with your actual API key

### Basic Configuration

The application uses a `.env` file for configuration. Here's a minimal setup:

```env
# Required: Your Google Gemini API Key
GOOGLE_API_KEY=AIzaSyBiTiTcY1Q8SxdG_d8TfjT6Tnlssw6jSCE

# Optional: Adjust these based on your needs
MAX_FILE_SIZE=50000000
MAX_CONVERSATION_HISTORY=10
DEFAULT_CHART_LIBRARY=plotly
```

## Environment Variables

### Required Variables

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `GOOGLE_API_KEY` | Google Gemini API key | `AIzaSy...` | Must start with 'AIza', 39 chars long |

### Application Limits

| Variable | Description | Default | Range |
|----------|-------------|---------|-------|
| `MAX_FILE_SIZE` | Maximum CSV file size (bytes) | `50000000` | 1MB - 500MB |
| `MAX_CONVERSATION_HISTORY` | Max conversation messages | `10` | 5 - 100 |
| `MAX_QUERY_LENGTH` | Max user query length | `1000` | 100 - 5000 |
| `MAX_RESPONSE_LENGTH` | Max AI response length | `10000` | 1000 - 50000 |

### Visualization Settings

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `DEFAULT_CHART_LIBRARY` | Chart library to use | `plotly` | `plotly`, `matplotlib` |
| `CHART_WIDTH` | Default chart width (px) | `800` | 400 - 1600 |
| `CHART_HEIGHT` | Default chart height (px) | `600` | 300 - 1200 |

### Performance Settings

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `PANDAS_CHUNKSIZE` | Chunk size for large files | `10000` | Adjust based on memory |
| `MEMORY_LIMIT_MB` | Memory limit (MB) | `1000` | System dependent |

### Development Settings

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `DEBUG_MODE` | Enable debug logging | `false` | `true`, `false` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENABLE_PROFILING` | Enable performance profiling | `false` | `true`, `false` |

### Security Settings

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `ENABLE_CODE_EXECUTION` | Allow AI code execution | `true` | Required for analysis |

### File Handling

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `TEMP_DIR` | Temporary files directory | `temp` | Relative to project root |

## Validation and Health Checks

### Startup Validation

Run the startup validator to check your configuration:

```bash
# Run comprehensive validation
python src/utils/startup_validator.py
```

The validator checks:
- Python version and virtual environment
- Project structure completeness
- Required dependencies
- Environment variables
- Configuration validity
- File system permissions
- API key format

### Manual Configuration Check

You can also check configuration programmatically:

```python
from src.config import ConfigurationManager

# Create configuration manager
config_manager = ConfigurationManager()

# Run startup validation
success, report = config_manager.perform_startup_validation()

# Check if configuration is healthy
is_healthy = config_manager.is_healthy()

print(f"Configuration healthy: {is_healthy}")
```

### Health Status Endpoint

Get detailed health status:

```python
from src.config import Config

config = Config()
health_status = config.get_health_status()
print(health_status)
```

## Troubleshooting

### Common Issues

#### 1. API Key Issues

**Problem**: "GOOGLE_API_KEY is required but not set"

**Solutions**:
- Ensure `.env` file exists in project root
- Verify API key is correctly set in `.env`
- Check API key format (should start with 'AIza')
- Restart the application after changing `.env`

#### 2. Import Errors

**Problem**: "ModuleNotFoundError: No module named 'streamlit'"

**Solutions**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

#### 3. File Upload Issues

**Problem**: "File size exceeds maximum allowed"

**Solutions**:
- Increase `MAX_FILE_SIZE` in `.env`
- Compress or reduce your CSV file size
- Check available system memory

#### 4. Memory Issues

**Problem**: Application crashes with large files

**Solutions**:
- Reduce `PANDAS_CHUNKSIZE` in `.env`
- Increase `MEMORY_LIMIT_MB` if system allows
- Use smaller datasets for testing

#### 5. Visualization Issues

**Problem**: Charts not displaying correctly

**Solutions**:
- Try switching `DEFAULT_CHART_LIBRARY` to `matplotlib`
- Adjust `CHART_WIDTH` and `CHART_HEIGHT`
- Check browser compatibility for Plotly charts

### Debug Mode

Enable debug mode for detailed logging:

```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

This provides:
- Detailed error messages
- Configuration validation details
- API request/response logging
- Performance metrics

### Validation Report

Generate a detailed validation report:

```bash
python src/utils/startup_validator.py
```

This creates a JSON report with:
- All validation check results
- Detailed error messages
- Specific suggestions for fixes
- System environment information

## Advanced Configuration

### Custom Configuration Files

Load configuration from custom `.env` file:

```python
from src.config import Config

# Load from custom file
config = Config.create_from_env_file('custom.env')
```

### Configuration Export

Save current configuration:

```python
from src.config import Config

config = Config()
# Save without sensitive data
config.save_to_env_file('config_backup.env', include_sensitive=False)
# Save with sensitive data (be careful!)
config.save_to_env_file('config_full.env', include_sensitive=True)
```

### Performance Tuning

For large datasets:

```env
# Increase memory limits
MEMORY_LIMIT_MB=2000
PANDAS_CHUNKSIZE=5000

# Optimize for performance
ENABLE_PROFILING=true
```

For development:

```env
# Enable all debugging
DEBUG_MODE=true
LOG_LEVEL=DEBUG
ENABLE_PROFILING=true

# Smaller limits for testing
MAX_FILE_SIZE=10000000
MAX_CONVERSATION_HISTORY=5
```

### Security Hardening

For production environments:

```env
# Disable unnecessary features
DEBUG_MODE=false
ENABLE_PROFILING=false
LOG_LEVEL=WARNING

# Restrict capabilities
MAX_FILE_SIZE=25000000
MAX_QUERY_LENGTH=500
ENABLE_CODE_EXECUTION=true  # Required for analysis
```

## Running the Application

After configuration:

```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run startup validation (optional but recommended)
python src/utils/startup_validator.py

# Start the application
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

## Configuration Validation Checklist

Before running the application, ensure:

- [ ] Python 3.8+ is installed
- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed (`pip install -r requirements.txt`)
- [ ] `.env` file exists with required variables
- [ ] Google API key is valid and has proper permissions
- [ ] Startup validation passes (`python src/utils/startup_validator.py`)
- [ ] Temp directory is writable
- [ ] System has adequate memory for your datasets

## Support

If you encounter issues:

1. Run the startup validator for diagnostic information
2. Check the troubleshooting section above
3. Enable debug mode for detailed logging
4. Review the validation report for specific guidance
5. Ensure all prerequisites are met

For additional help, provide:
- Validation report output
- Error messages with full stack traces
- System information (OS, Python version)
- Configuration details (without sensitive data)
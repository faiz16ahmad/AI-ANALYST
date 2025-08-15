---
inclusion: manual
---

# CSV Data Analyst Technical Standards

## Technology Stack Requirements

### Python Version
- **Required**: Python 3.11 or higher
- All code must be compatible with Python 3.11+ features and syntax

### Frontend Framework
- **Primary Framework**: Streamlit
- All user interface components must be built using Streamlit
- Follow Streamlit best practices for session state management and component organization

### AI Orchestration
- **Framework**: LangChain
- All AI-related logic must be orchestrated through LangChain
- Use LangChain's agent framework for data analysis tasks

### Large Language Model
- **Provider**: Google Gemini
- **Model**: gemini-1.5-flash-latest
- All LLM interactions must use the specified Gemini model through LangChain's ChatGoogleGenerativeAI

### Data Analysis
- **Library**: pandas
- All data manipulation and analysis must use pandas DataFrames
- Leverage pandas' built-in statistical and analytical functions

### Visualization
- **Library**: matplotlib
- All charts and graphs must be generated using matplotlib
- Charts should be optimized for display within Streamlit interface

## Security and Configuration Standards

### Environment Variables
- **API Keys**: All API keys must be stored as environment variables
- **Configuration File**: Use .env file for local development
- **Access Pattern**: Load environment variables using python-dotenv or similar

### Version Control
- **Git Ignore**: Create .gitignore file that excludes:
  - .env files
  - __pycache__/ directories
  - *.pyc files
  - Virtual environment directories
  - IDE-specific files

### Required .gitignore Entries
```
.env
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.idea/
.vscode/
*.log
```

## Code Organization Standards

### Project Structure
- Use src/ directory for main application code
- Separate concerns into logical modules
- Follow Python package structure conventions

### Dependency Management
- Use requirements.txt for dependency specification
- Pin major versions for stability
- Include all necessary dependencies for the technology stack

### Error Handling
- Implement comprehensive error handling for all external API calls
- Provide user-friendly error messages in the Streamlit interface
- Log errors appropriately for debugging purposes
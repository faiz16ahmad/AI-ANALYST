# AI-ANALYST Project Structure & Documentation

## 🎯 Project Overview
**AI-ANALYST** is an intelligent CSV data analysis tool powered by Google Gemini AI and LangChain. It provides a conversational interface for analyzing CSV data using natural language queries and generates interactive visualizations.

## 🏗️ Project Architecture

### Core Technology Stack
- **Frontend**: Streamlit (Web Interface)
- **AI Engine**: Google Gemini 2.5 Flash-Lite via LangChain
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **AI Framework**: LangChain with custom tools
- **Configuration**: Environment variables with dotenv

## 📁 File Structure & Purpose

### Root Directory
```
DATA-AI/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── PROJECT_STRUCTURE.md         # This file - detailed project documentation
├── .env.example                 # Environment variables template
├── src/                         # Main source code directory
├── temp/                        # Temporary files directory
└── venv/                        # Python virtual environment
```

### Source Code (`src/`)

#### 1. **Main Application (`src/app.py`)**
- **Purpose**: Entry point and main Streamlit application
- **Key Functions**:
  - Initializes the Streamlit web interface
  - Manages session state and user interactions
  - Coordinates between UI components and AI agent
  - Handles file uploads and data processing
  - Manages conversation flow and visualization display
- **Dependencies**: All other modules in the project
- **Entry Point**: `streamlit run src/app.py`

#### 2. **Configuration Management (`src/config.py`)**
- **Purpose**: Centralized configuration and environment management
- **Key Features**:
  - Environment variable loading and validation
  - API key management for Google Gemini
  - Application limits and settings
  - Visualization preferences
  - Security and performance configurations
- **Classes**:
  - `Config`: Main configuration dataclass
  - `ConfigurationManager`: Runtime configuration management
  - `config_manager`: Global configuration instance

#### 3. **AI Agent System (`src/agents/csv_analyst_agent.py`)**
- **Purpose**: Core AI intelligence for data analysis
- **Key Components**:
  - `CSVAnalystAgent`: Main agent class with Gemini integration
  - `AgentManager`: Manages agent lifecycle and configuration
  - `AgentResponse`: Structured response format
- **Capabilities**:
  - Natural language query processing
  - Pandas DataFrame analysis
  - Integration with visualization tools
  - Conversation memory management
- **AI Model**: Google Gemini 2.5 Flash-Lite

#### 4. **User Interface Components (`src/components/conversation_ui.py`)**
- **Purpose**: Streamlit UI components for conversation management
- **Key Classes**:
  - `ConversationManager`: Manages conversation state and flow
  - `ChatInterface`: Handles user input and chat display
  - `ConversationDisplay`: Renders conversation history
  - `ConversationMessage`: Data structure for messages
- **Features**:
  - Real-time chat interface
  - Message formatting and styling
  - Conversation history management
  - Expandable message content

#### 5. **Utility Modules (`src/utils/`)**

##### **Visualization (`src/utils/visualization.py`)**
- **Purpose**: Chart generation and visualization management
- **Key Features**:
  - Support for multiple chart types (bar, line, scatter, histogram, box, pie, heatmap, area, regression, treemap)
  - Dual library support (Plotly and Matplotlib)
  - Natural language query parsing for visualization requests
  - Chart customization and styling
- **Classes**:
  - `VisualizationTool`: Main visualization engine
  - `VisualizationParser`: Parses natural language for chart requests
  - `ChartType`: Enumeration of supported chart types

##### **LangChain Integration (`src/utils/langchain_viz_tool.py`)**
- **Purpose**: Custom LangChain tool for visualization generation
- **Key Features**:
  - Integrates visualization capabilities with LangChain agents
  - Handles complex visualization requests
  - Provides structured input/output for AI agent
- **Classes**:
  - `DataFrameVisualizationTool`: LangChain tool for visualizations
  - `VisualizationInput`: Input schema for the tool

##### **Error Handling (`src/utils/error_handler.py`)**
- **Purpose**: Centralized error management and user-friendly messaging
- **Key Features**:
  - Categorized error handling (API, network, file, data, validation, etc.)
  - Severity-based error classification
  - Graceful degradation capabilities
  - User-friendly error messages with suggestions
- **Classes**:
  - `ErrorHandler`: Main error handling system
  - `ValidationUtils`: Input and configuration validation
  - `GracefulDegradation`: Fallback mechanisms

##### **Package Validation (`src/utils/package_validator.py`)**
- **Purpose**: Security and compatibility validation for code execution
- **Key Features**:
  - Validates allowed/forbidden Python packages
  - Prevents unsafe code execution
  - Suggests alternatives for unavailable packages
- **Security**: Whitelist approach for package imports

##### **Startup Validation (`src/utils/startup_validator.py`)**
- **Purpose**: System health checks and dependency validation
- **Key Features**:
  - Validates required packages and versions
  - Checks system compatibility
  - Verifies API configurations
  - Ensures proper project setup

##### **Custom Python Tool (`src/utils/custom_python_tool.py`)**
- **Purpose**: Custom LangChain tool for Python code execution
- **Key Features**:
  - Safe Python code execution
  - Integration with pandas DataFrame operations
  - Custom tool for data analysis tasks

## 🔄 Data Flow & Component Interaction

### 1. **Application Startup**
```
app.py → config.py → startup_validator.py → csv_analyst_agent.py
```

### 2. **User Interaction Flow**
```
User Input → ChatInterface → ConversationManager → CSVAnalystAgent → VisualizationTool → Response Display
```

### 3. **Data Processing Pipeline**
```
CSV Upload → Pandas DataFrame → AI Agent Analysis → Visualization Generation → Streamlit Display
```

### 4. **Error Handling Flow**
```
Any Error → ErrorHandler → User-Friendly Message → Graceful Degradation (if possible)
```

## 🚀 Key Features & Capabilities

### **Natural Language Processing**
- Conversational interface for data analysis
- Context-aware responses using conversation memory
- Multi-turn conversations with data context

### **Data Analysis**
- Pandas DataFrame operations via AI agent
- Statistical analysis and insights
- Data validation and cleaning suggestions

### **Visualization Generation**
- 10+ chart types supported
- Automatic chart type detection from natural language
- Interactive Plotly charts with Streamlit integration
- Fallback to Matplotlib for complex visualizations

### **Security & Safety**
- Package validation for code execution
- API key management and validation
- File size and type restrictions
- Graceful error handling and degradation

## 🔧 Configuration & Environment

### **Required Environment Variables**
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### **Optional Environment Variables**
```bash
DEFAULT_CHART_LIBRARY=plotly          # Default visualization library
MAX_FILE_SIZE=50000000               # Maximum file size in bytes
DEBUG_MODE=false                     # Enable debug logging
LOG_LEVEL=INFO                       # Logging level
```

## 📊 Supported Data Operations

### **Chart Types**
- **Basic**: Bar, Line, Scatter, Histogram, Box, Pie
- **Advanced**: Heatmap, Area, Regression, Treemap
- **Complex**: Dual-axis, Combination charts, Multi-panel

### **Data Analysis**
- Descriptive statistics
- Correlation analysis
- Data distribution analysis
- Trend analysis
- Grouping and aggregation

## 🛠️ Development & Maintenance

### **Adding New Features**
1. **New Chart Types**: Add to `ChartType` enum in `visualization.py`
2. **New Tools**: Create in `utils/` and integrate with agent
3. **UI Components**: Add to `components/` directory
4. **Configuration**: Extend `Config` class in `config.py`

### **Testing**
- Run health check: `python health_check.py`
- Test individual components in isolation
- Validate configuration changes

### **Dependencies**
- All dependencies listed in `requirements.txt`
- Virtual environment management via `venv/`
- Version compatibility documented in requirements

## 🔍 Troubleshooting

### **Common Issues**
1. **API Key Errors**: Check `GOOGLE_API_KEY` environment variable
2. **Import Errors**: Verify virtual environment activation
3. **File Upload Issues**: Check file size and format restrictions
4. **Visualization Errors**: Verify chart library dependencies

### **Debug Mode**
Enable debug mode by setting `DEBUG_MODE=true` in environment variables for detailed logging.

## 📈 Project Evolution

This documentation should be updated whenever:
- New files or components are added
- Existing functionality is modified
- Dependencies or requirements change
- New features or capabilities are implemented
- Architecture changes occur

---

**Last Updated**: [Current Date]
**Project Version**: Based on current codebase
**Maintainer**: [Your Name/Team]

For detailed technical implementation, refer to individual file docstrings and inline comments.

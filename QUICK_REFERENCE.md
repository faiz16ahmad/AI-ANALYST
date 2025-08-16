# AI-ANALYST Quick Reference

## 🚀 Quick Start
```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run
streamlit run src/app.py
```

## 📁 File Quick Reference

### **Core Files**
| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `src/app.py` | Main Streamlit app | Entry point, UI coordination |
| `src/config.py` | Configuration management | `Config`, `ConfigurationManager` |
| `src/agents/csv_analyst_agent.py` | AI agent | `CSVAnalystAgent`, `AgentManager` |

### **UI Components**
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/components/conversation_ui.py` | Chat interface | `ConversationManager`, `ChatInterface` |

### **Utilities**
| File | Purpose | Key Features |
|------|---------|--------------|
| `src/utils/visualization.py` | Chart generation | 10+ chart types, Plotly/Matplotlib |
| `src/utils/langchain_viz_tool.py` | LangChain integration | Custom visualization tool |
| `src/utils/error_handler.py` | Error management | Categorized errors, user-friendly messages |
| `src/utils/package_validator.py` | Security | Package whitelist, safe execution |
| `src/utils/startup_validator.py` | Health checks | Dependency validation, system checks |

## 🔑 Environment Variables
```bash
GOOGLE_API_KEY=your_key_here          # Required
DEFAULT_CHART_LIBRARY=plotly          # Optional
MAX_FILE_SIZE=50000000                # Optional (50MB default)
DEBUG_MODE=false                      # Optional
```

## 📊 Supported Chart Types
- **Basic**: Bar, Line, Scatter, Histogram, Box, Pie
- **Advanced**: Heatmap, Area, Regression, Treemap
- **Complex**: Dual-axis, Combination, Multi-panel

## 🔄 Data Flow
1. **Upload CSV** → Pandas DataFrame
2. **User Query** → AI Agent (Gemini)
3. **Analysis** → Pandas operations + Visualization
4. **Response** → Text + Interactive charts

## 🛠️ Development
- **New Charts**: Add to `ChartType` enum in `visualization.py`
- **New Tools**: Create in `utils/` and integrate with agent
- **New UI**: Add to `components/` directory
- **Config**: Extend `Config` class in `config.py`

## 🐛 Common Issues
- **API Key**: Check `GOOGLE_API_KEY` environment variable
- **Imports**: Verify virtual environment activation
- **File Size**: Check `MAX_FILE_SIZE` setting
- **Debug**: Set `DEBUG_MODE=true` for detailed logs

## 📚 Dependencies
- **AI**: `langchain`, `langchain-google-genai`
- **Data**: `pandas`, `numpy`, `scikit-learn`
- **Viz**: `plotly`, `matplotlib`
- **Web**: `streamlit`
- **Config**: `python-dotenv`

---
**For detailed documentation**: See `PROJECT_STRUCTURE.md`
**For setup issues**: Check `src/utils/startup_validator.py`
**For errors**: Check `src/utils/error_handler.py`

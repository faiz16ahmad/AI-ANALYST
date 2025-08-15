# AI-ANALYST

An intelligent CSV data analysis tool powered by Google Gemini AI and LangChain, featuring natural language queries and interactive visualizations.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **AI-Powered Analysis**: Uses Google Gemini 2.5 Flash-Lite for intelligent data insights
- **Interactive Visualizations**: Automatic chart generation with Plotly and Matplotlib
- **Multiple Chart Types**: Bar charts, line charts, scatter plots, histograms, box plots, pie charts, heatmaps, and regression plots
- **Streamlit Web Interface**: User-friendly web application
- **CSV File Upload**: Easy drag-and-drop file upload
- **Conversation History**: Keep track of your analysis sessions

## 📋 Requirements

- Python 3.8+
- Google API Key (for Gemini AI)
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/faiz16ahmad/AI-ANALYST.git
   cd AI-ANALYST
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🚀 Usage

1. **Start the application**

   ```bash
   streamlit run src/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a CSV file** using the file uploader

4. **Ask questions** about your data in natural language, such as:
   - "Show me a histogram of age distribution"
   - "Create a scatter plot of price vs quantity"
   - "What's the correlation between sales and marketing spend?"
   - "Generate a box plot for Monthly_Allowance"

## 📊 Supported Visualizations

- **Bar Charts**: Compare categories and values
- **Line Charts**: Show trends over time
- **Scatter Plots**: Explore relationships between variables
- **Histograms**: Display data distributions
- **Box Plots**: Show statistical summaries
- **Pie Charts**: Display proportions
- **Heatmaps**: Visualize correlation matrices
- **Regression Plots**: Show linear relationships with trend lines

## 🏗️ Project Structure

```
AI-ANALYST/
├── src/
│   ├── agents/
│   │   └── csv_analyst_agent.py    # Main AI agent
│   ├── components/
│   │   └── conversation_ui.py      # UI components
│   ├── utils/
│   │   ├── visualization.py        # Chart generation
│   │   ├── langchain_viz_tool.py   # LangChain integration
│   │   └── error_handler.py        # Error handling
│   ├── config.py                   # Configuration management
│   └── app.py                      # Main Streamlit app
├── .kiro/
│   └── specs/                      # Project specifications
├── requirements.txt                # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🔧 Configuration

The application uses environment variables for configuration:

- `GOOGLE_API_KEY`: Your Google API key for Gemini AI
- `DEFAULT_CHART_LIBRARY`: Default visualization library (plotly/matplotlib)
- `MAX_FILE_SIZE_MB`: Maximum CSV file size in MB
- `STREAMLIT_THEME`: UI theme configuration

## 🧪 Testing

Run the health check to verify your setup:

```bash
python health_check.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI for natural language processing
- LangChain for AI agent framework
- Streamlit for the web interface
- Plotly and Matplotlib for visualizations

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ and AI**

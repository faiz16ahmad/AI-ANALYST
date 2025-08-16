"""
CSV Analyst Agent - LangChain agent with Google Gemini integration

This module implements the core agent functionality for analyzing CSV data
using natural language queries with Google Gemini 1.5 Flash model.
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

try:
    from ..config import Config
    from ..utils.langchain_viz_tool import create_visualization_tool
    from ..utils.visualization import VisualizationResult
    from ..utils.error_handler import error_handler
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import Config
    from utils.langchain_viz_tool import create_visualization_tool
    from utils.visualization import VisualizationResult
    from utils.error_handler import error_handler


@dataclass
class AgentResponse:
    """Response object from the CSV analyst agent"""
    text_response: str
    visualization: Optional[Any] = None
    execution_details: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None


class CSVAnalystAgent:
    """
    Main agent class for CSV data analysis using LangChain and Google Gemini.
    
    This agent uses ChatGoogleGenerativeAI with gemini-1.5-flash-latest model
    and create_pandas_dataframe_agent for data analysis capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CSV analyst agent.
        
        Args:
            api_key: Google API key for Gemini access. If None, will use config.
        """
        self.config = Config()
        self.api_key = api_key or self.config.GOOGLE_API_KEY
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the language model
        self.llm = None
        self.agent = None
        self.dataframe = None
        self.visualization_tool = None
        # Note: ConversationBufferMemory is deprecated, but still used for compatibility
        # with create_pandas_dataframe_agent. This will be updated in future versions.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=4000
        )
        
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the ChatGoogleGenerativeAI model with enhanced error handling."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",  # Upgraded to Gemini 2.5 Flash-Lite for optimal performance
                google_api_key=self.api_key,
                temperature=0.1,  # Low temperature for more consistent analysis
                # convert_system_message_to_human is deprecated, removing it
            )
        except Exception as e:
            error_info = error_handler.handle_error(e, "LLM initialization")
            raise RuntimeError(f"Failed to initialize Google Gemini model: {error_info.user_message}")
    
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load a pandas DataFrame for analysis.
        
        Args:
            df: The pandas DataFrame to analyze
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        self.dataframe = df.copy()
        
        # Create visualization tool
        self.visualization_tool = create_visualization_tool(
            self.dataframe, 
            default_library=self.config.DEFAULT_CHART_LIBRARY
        )
        
        # Create custom Python execution tool for capturing LLM-generated figures
        try:
            from ..utils.custom_python_tool import create_custom_python_tool
        except ImportError:
            from utils.custom_python_tool import create_custom_python_tool
        
        self.python_tool = create_custom_python_tool(self.dataframe)
        
        # Create the pandas DataFrame agent with visualization tool
        try:
            # Create pandas DataFrame agent with proper configuration for Gemini
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.dataframe,
                verbose=True,
                agent_type="zero-shot-react-description",  # Better compatibility with Gemini
                allow_dangerous_code=True,  # Required for pandas agent
                max_iterations=10,  # Increased iterations for complex queries
                handle_parsing_errors=True,  # Handle parsing errors gracefully
                extra_tools=[self.visualization_tool, self.python_tool]  # Add visualization and Python tools
            )
        except Exception as e:
            error_info = error_handler.handle_error(e, "Pandas DataFrame agent creation")
            raise RuntimeError(f"Failed to create pandas DataFrame agent: {error_info.user_message}")
    
    def process_query(self, question: str) -> AgentResponse:
        """
        Process a natural language query about the loaded DataFrame.
        
        Args:
            question: Natural language question about the data
            
        Returns:
            AgentResponse object with the analysis results
        """
        if not self.agent:
            return AgentResponse(
                text_response="No data loaded. Please upload a CSV file first.",
                success=False,
                error_message="No DataFrame loaded"
            )
        
        if not question.strip():
            return AgentResponse(
                text_response="Please provide a question about your data.",
                success=False,
                error_message="Empty question"
            )
        
        try:
            # Add context about the dataset to the question
            enhanced_question = self._enhance_question_with_context(question)
            
            # Process the query through the agent
            start_time = datetime.now()
            result = self.agent.invoke({"input": enhanced_question})
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract the response
            response_text = result.get("output", "No response generated")
            
            # Check if a visualization was created
            visualization = None
            if self.visualization_tool:
                visualization = self.visualization_tool.get_current_visualization()
            
            # Also check for figures from custom Python tool
            if not visualization and self.python_tool:
                visualization = self.python_tool.get_last_figure()
                if visualization:
                    # Clear the figure after capturing it
                    self.python_tool.clear_figure()
            
            return AgentResponse(
                text_response=response_text,
                visualization=visualization,
                execution_details={
                    "execution_time": execution_time,
                    "question": question,
                    "enhanced_question": enhanced_question,
                    "timestamp": datetime.now().isoformat(),
                    "has_visualization": visualization is not None
                },
                success=True
            )
            
        except Exception as e:
            error_str = str(e)
            
            # Check for package-related errors
            if any(pkg in error_str.lower() for pkg in ['modulenotfounderror', 'importerror', 'no module named']):
                # This is likely a missing package error
                package_error_response = f"""
I encountered an error trying to import a required package. This might be because the code attempted to use a package that's not installed in this environment.

Available packages: pandas, plotly, numpy
Error: {error_str}

Let me try a different approach using only the available packages.
"""
                return AgentResponse(
                    text_response=package_error_response,
                    success=False,
                    error_message=f"Package import error: {error_str}",
                    execution_details={
                        "question": question,
                        "error": str(e),
                        "error_type": "package_import_error",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Use enhanced error handling for other errors
            error_info = error_handler.handle_error(e, "Agent query processing")
            return AgentResponse(
                text_response=error_info.user_message,
                success=False,
                error_message=str(e),
                execution_details={
                    "question": question,
                    "error": str(e),
                    "error_category": error_info.category.value,
                    "error_suggestions": error_info.suggestions,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _enhance_question_with_context(self, question: str) -> str:
        """
        Enhance the user question with context about the dataset.
        
        Args:
            question: Original user question
            
        Returns:
            Enhanced question with dataset context
        """
        if self.dataframe is None:
            return question
        
        # Get basic info about the dataset
        shape_info = f"The dataset has {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns."
        columns_info = f"The columns are: {', '.join(self.dataframe.columns.tolist())}"
        
        # Add data types info for key columns (first 5)
        dtypes_info = ""
        if len(self.dataframe.columns) > 0:
            key_columns = self.dataframe.columns[:5]
            dtype_details = []
            for col in key_columns:
                dtype_details.append(f"{col} ({self.dataframe[col].dtype})")
            dtypes_info = f"Key column types: {', '.join(dtype_details)}"
        
        # Check if this is a visualization request and handle it specially
        is_heatmap_request = any(word in question.lower() for word in ['heatmap', 'heat map'])
        is_viz_request = any(word in question.lower() for word in ['plot', 'chart', 'graph', 'visualize', 'show', 'display'])
        
        if is_heatmap_request:
            enhanced_question = f"""
Dataset Context:
{shape_info}
{columns_info}
{dtypes_info}

User Question: {question}

This is a heatmap visualization request. You MUST use the create_visualization tool to create this heatmap. 

Step 1: Call create_visualization("{question}")
Step 2: Provide analysis of the data shown in the heatmap

DO NOT attempt to create heatmaps using pandas, matplotlib, seaborn, or any other method. The create_visualization tool handles all heatmap types including pivot table heatmaps and correlation matrices.

IMPORTANT: If you need to write Python code, ALWAYS use execute_python_code instead of python_repl_ast for better Streamlit integration.
"""
        elif is_viz_request:
            enhanced_question = f"""
Dataset Context:
{shape_info}
{columns_info}
{dtypes_info}

User Question: {question}

This is a visualization request. Follow this SMART ROUTING process:

Step 1: Call create_visualization("{question}")
Step 2: Analyze the response:
   - If successful: Great! Provide insights about the visualization
   - If "CUSTOM_CODE_REQUIRED": 
     IMMEDIATELY write custom code using execute_python_code

IMPORTANT: When you see "CUSTOM_CODE_REQUIRED", this means:
1. Our simple tool cannot handle this request
2. You MUST write pandas/plotly code using execute_python_code (NOT python_repl_ast)
3. DO NOT keep trying create_visualization - it will keep failing
4. This overrides the "don't use pandas directly" rule
5. ALWAYS prefer execute_python_code over python_repl_ast for better Streamlit integration

WHEN TO WRITE CUSTOM CODE (using execute_python_code):
- Any time create_visualization returns "CUSTOM_CODE_REQUIRED"
- Complex styling (gradients, custom colors, annotations)
- ALWAYS use execute_python_code instead of python_repl_ast for Python code execution
- Multiple chart types combined
- Advanced features (dual-axis, subplots, custom layouts)
- Ambiguous requests that need data exploration

CODE EXAMPLES FOR COMMON CASES:

DUAL-AXIS CHART:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Aggregate data by time period
monthly_data = df.groupby(df['OrderDate'].dt.to_period('M'))[['TotalSales', 'Profit']].sum().reset_index()
monthly_data['Month'] = monthly_data['OrderDate'].astype(str)

# Create dual-axis chart
fig = make_subplots(specs=[[{{"secondary_y": True}}]])
fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data['TotalSales'], name="Sales"), secondary_y=False)
fig.add_trace(go.Scatter(x=monthly_data['Month'], y=monthly_data['Profit'], mode='lines+markers', name="Profit"), secondary_y=True)
fig.update_yaxes(title_text="Sales", secondary_y=False)
fig.update_yaxes(title_text="Profit", secondary_y=True)
fig.update_layout(title="Monthly Sales and Profit")
# Return the figure for Streamlit display
fig
```

CUSTOM STYLING:
```python
import plotly.express as px

# Create chart with custom styling
fig = px.bar(df, x='Category', y='Sales', color='Sales',
             color_continuous_scale='Blues',
             title='Sales by Category')
fig.update_traces(texttemplate='%{{y}}', textposition='outside')
# Return the figure for Streamlit display
fig
```

AVAILABLE PACKAGES: Only use these pre-installed packages:
- pandas (as pd) - already imported as 'df'
- plotly.express (as px) 
- plotly.graph_objects (as go)
- plotly.subplots (make_subplots)
- numpy (as np)

DO NOT use: seaborn, matplotlib.pyplot, sklearn, or any other packages.

IMPORTANT: When creating visualizations with code:
- DO NOT use fig.show() - this opens new browser tabs
- Instead, end your code with just 'fig' to return the figure object
- The figure will be automatically displayed in the Streamlit interface

HEATMAP CODE EXAMPLE (for when create_visualization fails):
```python
import plotly.express as px

# Create pivot table
pivot_data = df.pivot_table(values='Profit', index='Category', columns='Region', aggfunc='sum')

# Create heatmap
fig = px.imshow(pivot_data.values, 
                x=pivot_data.columns, 
                y=pivot_data.index,
                text_auto=True,
                title="Profit by Category and Region")
# Return the figure for Streamlit display
fig
```

PRINCIPLE: Try the tool first for simple cases, write code for complex/custom requirements using ONLY the available packages listed above.
"""
        else:
            enhanced_question = f"""
Dataset Context:
{shape_info}
{columns_info}
{dtypes_info}

User Question: {question}

Please provide a comprehensive analysis based on the user's question. If you need to create any visualizations, use the create_visualization tool.

IMPORTANT: If you need to write Python code, ALWAYS use execute_python_code instead of python_repl_ast for better Streamlit integration.
"""
        
        return enhanced_question
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history from memory.
        
        Returns:
            List of conversation messages
        """
        if not self.memory:
            return []
        
        messages = []
        try:
            chat_history = self.memory.chat_memory.messages
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    messages.append({
                        "role": "user",
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
                elif isinstance(message, AIMessage):
                    messages.append({
                        "role": "assistant", 
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
        
        return messages
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        if self.memory:
            self.memory.clear()
    
    def get_dataframe_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded DataFrame.
        
        Returns:
            Dictionary with DataFrame information or None if no data loaded
        """
        if self.dataframe is None:
            return None
        
        return {
            "shape": self.dataframe.shape,
            "columns": self.dataframe.columns.tolist(),
            "dtypes": self.dataframe.dtypes.to_dict(),
            "null_counts": self.dataframe.isnull().sum().to_dict(),
            "memory_usage": f"{self.dataframe.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "sample_data": self.dataframe.head().to_dict()
        }
    
    def is_ready(self) -> bool:
        """
        Check if the agent is ready to process queries.
        
        Returns:
            True if agent is initialized and has data loaded
        """
        return self.agent is not None and self.dataframe is not None


class AgentManager:
    """
    Manager class for handling agent lifecycle and configuration.
    """
    
    def __init__(self):
        self.agent = None
        self.config = Config()
    
    def initialize_agent(self, api_key: Optional[str] = None) -> CSVAnalystAgent:
        """
        Initialize a new CSV analyst agent with enhanced error handling.
        
        Args:
            api_key: Optional Google API key
            
        Returns:
            Initialized CSVAnalystAgent instance
        """
        try:
            self.agent = CSVAnalystAgent(api_key=api_key)
            return self.agent
        except Exception as e:
            error_info = error_handler.handle_error(e, "Agent manager initialization")
            raise RuntimeError(f"Failed to initialize agent: {error_info.user_message}")
    
    def get_agent(self) -> Optional[CSVAnalystAgent]:
        """Get the current agent instance."""
        return self.agent
    
    def validate_configuration(self) -> tuple[bool, str]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.GOOGLE_API_KEY:
            return False, "Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
        
        return True, ""
    
    def reset_agent(self) -> None:
        """Reset the current agent instance."""
        if self.agent:
            self.agent.clear_memory()
        self.agent = None
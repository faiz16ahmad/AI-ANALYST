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

from ..config import Config
from ..utils.langchain_viz_tool import create_visualization_tool
from ..utils.visualization import VisualizationResult
from ..utils.error_handler import error_handler


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
                extra_tools=[self.visualization_tool]  # Add visualization tool
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
            # Use enhanced error handling
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
        
        enhanced_question = f"""
Dataset Context:
{shape_info}
{columns_info}
{dtypes_info}

User Question: {question}

Please provide a comprehensive analysis based on the user's question. If the question involves creating visualizations (charts, plots, graphs), use the create_visualization tool to generate the appropriate chart. You have access to various chart types including bar charts, line charts, scatter plots, histograms, box plots, pie charts, heatmaps, and area charts.

For visualization requests, call the create_visualization tool with a clear description of what to visualize, and then provide additional context and insights about the data in your response.
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
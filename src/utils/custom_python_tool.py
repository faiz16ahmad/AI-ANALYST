"""
Custom Python execution tool that can capture Plotly figures
"""

from typing import Optional, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from .visualization import VisualizationResult


class PythonCodeInput(BaseModel):
    """Input schema for the Python code execution tool"""
    code: str = Field(description="Python code to execute")


class CustomPythonTool(BaseTool):
    """
    Custom Python execution tool that can capture Plotly figures and integrate with our visualization system.
    """
    
    name: str = "execute_python_code"
    description: str = """
    PREFERRED PYTHON TOOL: Execute Python code and automatically capture Plotly figures for Streamlit display.
    
    Use this tool instead of python_repl_ast for ALL Python code execution, especially for:
    - Creating visualizations with pandas and plotly
    - Data analysis and manipulation
    - Any Python code that generates charts or figures
    
    This tool automatically captures and integrates Plotly figures with the Streamlit interface.
    Available packages: pandas (as pd), plotly.express (as px), plotly.graph_objects (as go), 
    plotly.subplots (make_subplots), numpy (as np)
    
    IMPORTANT: Always use this tool for Python code. End your code with the figure object (e.g., 'fig') to return it.
    """
    
    args_schema: type[BaseModel] = PythonCodeInput
    return_direct: bool = False
    
    def __init__(self, dataframe: pd.DataFrame, **kwargs):
        """
        Initialize the custom Python tool.
        
        Args:
            dataframe: The pandas DataFrame to make available in the execution environment
        """
        super().__init__(**kwargs)
        # Store dataframe as private attribute
        object.__setattr__(self, 'dataframe', dataframe)
        object.__setattr__(self, '_last_figure', None)
    
    def _run(self, code: str, **kwargs) -> str:
        """
        Execute Python code and capture any figure results.
        
        Args:
            code: Python code to execute
            
        Returns:
            String description of execution result
        """
        try:
            # Clear any previous figure
            object.__setattr__(self, '_last_figure', None)
            
            # Set up execution environment
            exec_globals = {
                'pd': pd,
                'df': self.dataframe,
                'np': None,  # Will be imported if needed
                'px': None,  # Will be imported if needed
                'go': None,  # Will be imported if needed
                'make_subplots': None,  # Will be imported if needed
            }
            
            # Import packages as needed
            try:
                import numpy as np
                exec_globals['np'] = np
            except ImportError:
                pass
            
            try:
                import plotly.express as px
                exec_globals['px'] = px
            except ImportError:
                pass
            
            try:
                import plotly.graph_objects as go
                exec_globals['go'] = go
            except ImportError:
                pass
            
            try:
                from plotly.subplots import make_subplots
                exec_globals['make_subplots'] = make_subplots
            except ImportError:
                pass
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec_locals = {}
                exec(code, exec_globals, exec_locals)
                
                # Try to get the result from the last expression
                # Look for common variable names that might contain the result
                result = None
                for var_name in ['fig', '_', 'result', 'chart']:
                    if var_name in exec_locals:
                        result = exec_locals[var_name]
                        break
            
            # Check if result is a Plotly figure
            if result is not None and hasattr(result, 'show'):
                # This looks like a Plotly figure
                object.__setattr__(self, '_last_figure', result)
                return "Python code executed successfully. Plotly figure created and captured."
            else:
                # No figure returned, but code executed
                stdout_output = stdout_capture.getvalue()
                if stdout_output:
                    return f"Python code executed successfully. Output: {stdout_output}"
                else:
                    return "Python code executed successfully."
                    
        except Exception as e:
            return f"Error executing Python code: {str(e)}"
    
    async def _arun(self, code: str, **kwargs) -> str:
        """Async version of _run"""
        return self._run(code, **kwargs)
    
    def get_last_figure(self) -> Optional[VisualizationResult]:
        """
        Get the last figure created by the Python code execution.
        
        Returns:
            VisualizationResult object or None if no figure was created
        """
        if self._last_figure is not None:
            return VisualizationResult(
                chart_data=self._last_figure,
                chart_type="custom",
                library="plotly",
                title="Custom Chart",
                success=True
            )
        return None
    
    def clear_figure(self) -> None:
        """Clear the stored figure"""
        object.__setattr__(self, '_last_figure', None)


def create_custom_python_tool(dataframe: pd.DataFrame) -> CustomPythonTool:
    """
    Factory function to create a custom Python execution tool.
    
    Args:
        dataframe: pandas DataFrame to make available in the execution environment
        
    Returns:
        CustomPythonTool instance
    """
    return CustomPythonTool(dataframe=dataframe)
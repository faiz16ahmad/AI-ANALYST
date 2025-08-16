"""
LangChain visualization tool for pandas DataFrame agent

This module provides a custom LangChain tool that enables the pandas DataFrame agent
to generate visualizations based on natural language requests.
"""

from typing import Optional, Any, Dict
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

from .visualization import VisualizationTool, VisualizationResult


class VisualizationInput(BaseModel):
    """Input schema for the visualization tool"""
    query: str = Field(description="Natural language description of the desired visualization")
    chart_type: Optional[str] = Field(default=None, description="Specific chart type (bar, line, scatter, histogram, box, pie, heatmap, area)")
    x_column: Optional[str] = Field(default=None, description="Column name for x-axis")
    y_column: Optional[str] = Field(default=None, description="Column name for y-axis")
    title: Optional[str] = Field(default=None, description="Chart title")


class DataFrameVisualizationTool(BaseTool):
    """
    Custom LangChain tool for creating visualizations from pandas DataFrames.
    
    This tool can be added to the pandas DataFrame agent to enable visualization
    capabilities through natural language queries.
    """
    
    name: str = "create_visualization"
    description: str = """
    Create visualizations (charts, plots, graphs) from the DataFrame based on natural language descriptions.
    
    Use this tool when the user asks for:
    - Charts, plots, or graphs
    - Visual representations of data
    - Histograms, bar charts, line charts, scatter plots, etc.
    - Data distributions or correlations
    - Heatmaps, correlation matrices, pivot table visualizations
    
    IMPORTANT: This tool handles ALL visualization types including heatmaps. Do NOT use pandas, matplotlib, seaborn, or other libraries directly.
    
    Input should be a natural language description of what to visualize.
    Examples:
    - "Create a bar chart of sales by region"
    - "Show a histogram of age distribution"
    - "Plot price vs quantity as a scatter plot"
    - "Generate a line chart showing trends over time"
    - "Show a heatmap of profit by category and region"
    - "Create a correlation matrix heatmap"
    """
    
    args_schema: type[BaseModel] = VisualizationInput
    return_direct: bool = False
    
    def __init__(self, dataframe: pd.DataFrame, default_library: str = "plotly", **kwargs):
        """
        Initialize the visualization tool.
        
        Args:
            dataframe: The pandas DataFrame to visualize
            default_library: Default visualization library ("plotly" or "matplotlib")
        """
        super().__init__(**kwargs)
        # Store dataframe and tools as private attributes to avoid Pydantic field issues
        object.__setattr__(self, 'dataframe', dataframe)
        object.__setattr__(self, 'viz_tool', VisualizationTool(default_library))
        object.__setattr__(self, '_current_visualization', None)
    
    def _run(self, query: str, chart_type: Optional[str] = None, 
             x_column: Optional[str] = None, y_column: Optional[str] = None,
             title: Optional[str] = None, **kwargs) -> str:
        """
        Execute the visualization tool.
        
        Args:
            query: Natural language description of visualization
            chart_type: Optional specific chart type
            x_column: Optional x-axis column
            y_column: Optional y-axis column
            title: Optional chart title
            
        Returns:
            String description of the created visualization
        """
        try:
            query_lower = query.lower()
            
            # BOX PLOT DETECTION AND ENHANCEMENT
            box_plot_indicators = ['box plot', 'boxplot', 'box chart']
            is_box_plot_request = any(indicator in query_lower for indicator in box_plot_indicators)
            
            # HEATMAP DETECTION AND ENHANCEMENT
            heatmap_indicators = ['heatmap', 'heat map', 'correlation matrix']
            is_heatmap_request = any(indicator in query_lower for indicator in heatmap_indicators)
            
            # REGRESSION DETECTION
            regression_indicators = [
                'regression', 'linear regression', 'regression line', 'regression plot',
                'trend line', 'trendline', 'trend', 'best fit', 'fit line', 'line of best fit',
                'linear relationship', 'linear correlation', 'correlation line',
                'make a line', 'draw a line', 'add a line', 'show the line',
                'line based on', 'line through', 'line connecting',
                'linear model', 'predictive line', 'prediction line'
            ]
            is_regression_request = any(indicator in query_lower for indicator in regression_indicators)
            
            # Additional context clues for regression
            context_clues = [
                'relationship between', 'correlation between', 'how does', 'predict',
                'linear', 'slope', 'intercept', 'r-squared', 'r²'
            ]
            has_context_clues = any(clue in query_lower for clue in context_clues)
            
            # Enhanced query processing
            if is_heatmap_request:
                print(f"🔥 Detected heatmap request: {query}")
                # For heatmaps, pass the original query directly
                enhanced_query = query
            elif is_box_plot_request:
                print(f"📊 Detected box plot request: {query}")
                # For box plots, ensure we extract the column name properly
                enhanced_query = query
                # Add explicit column guidance if not already specified
                if not x_column and not y_column:
                    # Try to extract column name from the query
                    column_keywords = ['for', 'of', 'show', 'display', 'plot']
                    for keyword in column_keywords:
                        if keyword in query_lower:
                            # Look for column name after the keyword
                            parts = query_lower.split(keyword)
                            if len(parts) > 1:
                                potential_column = parts[1].strip().split()[0] if parts[1].strip() else None
                                if potential_column:
                                    # Check if this matches any DataFrame column
                                    for col in self.dataframe.columns:
                                        if potential_column.lower() in col.lower() or col.lower() in potential_column.lower():
                                            enhanced_query += f" using column {col}"
                                            break
                                    break
            elif is_regression_request or (has_context_clues and ('line' in query_lower or 'plot' in query_lower)):
                print(f"🎯 Detected regression request: {query}")
                # Force regression chart creation
                enhanced_query = f"Create a regression plot with prominent red regression line and equation showing the relationship"
                if x_column:
                    enhanced_query += f" between {x_column}"
                if y_column:
                    enhanced_query += f" and {y_column}"
                enhanced_query += " with R-squared value displayed"
            else:
                # Regular query processing
                enhanced_query = query
                if chart_type:
                    enhanced_query += f" as a {chart_type} chart"
                if x_column:
                    enhanced_query += f" with {x_column} on x-axis"
                if y_column:
                    enhanced_query += f" and {y_column} on y-axis"
                
                # Add color guidance for categorical data
                if any(word in query_lower for word in ['color', 'green', 'red', 'placed', 'not placed']):
                    enhanced_query += " with appropriate colors for different categories"
            
            # Generate visualization
            viz_result = self.viz_tool.create_visualization(enhanced_query, self.dataframe)
            
            if viz_result and viz_result.success:
                # Store the visualization for later retrieval
                object.__setattr__(self, '_current_visualization', viz_result)
                
                # Return description of what was created
                description = f"Created a {viz_result.chart_type} chart"
                if viz_result.title:
                    description += f" titled '{viz_result.title}'"
                description += f" using {viz_result.library}."
                
                # Add information about the data being visualized
                if hasattr(viz_result, 'chart_data') and viz_result.chart_data is not None:
                    description += f" The visualization shows data from the DataFrame with {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns."
                
                return description
            else:
                error_msg = viz_result.error_message if viz_result else "Unknown error"
                
                # Check if this is a fallback request
                if error_msg and "FALLBACK_TO_CODE:" in error_msg:
                    return f"CUSTOM_CODE_REQUIRED. This visualization needs custom implementation. Create the chart using pandas and plotly with the specific requirements from the query. Use only available packages: pandas, plotly.express, plotly.graph_objects, numpy."
                
                return f"Failed to create visualization: {error_msg}"
                
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Async version of _run"""
        return self._run(query, **kwargs)
    
    def get_current_visualization(self) -> Optional[VisualizationResult]:
        """
        Get the most recently created visualization.
        
        Returns:
            VisualizationResult object or None
        """
        return self._current_visualization
    
    def clear_visualization(self) -> None:
        """Clear the current visualization"""
        object.__setattr__(self, '_current_visualization', None)
    
    def update_dataframe(self, new_dataframe: pd.DataFrame) -> None:
        """
        Update the DataFrame used for visualizations.
        
        Args:
            new_dataframe: New pandas DataFrame
        """
        object.__setattr__(self, 'dataframe', new_dataframe)
        self.clear_visualization()


def create_visualization_tool(dataframe: pd.DataFrame, default_library: str = "plotly") -> DataFrameVisualizationTool:
    """
    Factory function to create a visualization tool for a DataFrame.
    
    Args:
        dataframe: pandas DataFrame to visualize
        default_library: Default visualization library
        
    Returns:
        DataFrameVisualizationTool instance
    """
    return DataFrameVisualizationTool(dataframe=dataframe, default_library=default_library)
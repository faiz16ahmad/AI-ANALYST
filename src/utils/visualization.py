"""
Visualization utilities for CSV Data Analyst

This module provides visualization capabilities for the pandas DataFrame agent,
including chart generation with matplotlib and plotly, and integration with Streamlit.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Union, List, Tuple
import io
import base64
import re
from dataclasses import dataclass
from enum import Enum

# Import error handler for consistent error handling
try:
    from .error_handler import error_handler
except ImportError:
    # Fallback if error_handler is not available
    error_handler = None


class ChartType(Enum):
    """Supported chart types for visualization"""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    PIE = "pie"
    HEATMAP = "heatmap"
    AREA = "area"
    REGRESSION = "regression"  # Scatter plot with regression line


@dataclass
class VisualizationRequest:
    """Data class for visualization requests"""
    chart_type: ChartType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    additional_params: Dict[str, Any] = None


@dataclass
class VisualizationResult:
    """Result object for visualization generation"""
    chart_data: Any  # matplotlib figure or plotly figure
    chart_type: str
    library: str  # "matplotlib" or "plotly"
    title: str
    success: bool = True
    error_message: Optional[str] = None


class VisualizationParser:
    """Parser for extracting visualization requests from natural language queries"""
    
    # Keywords that indicate visualization requests
    VISUALIZATION_KEYWORDS = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'histogram', 'bar chart', 'line chart', 'scatter plot',
        'pie chart', 'box plot', 'heatmap', 'distribution'
    ]
    
    # Chart type patterns
    CHART_PATTERNS = {
        ChartType.BAR: [
            r'bar\s+chart', r'bar\s+plot', r'bar\s+graph',
            r'column\s+chart', r'vertical\s+bar'
        ],
        ChartType.LINE: [
            r'line\s+chart', r'line\s+plot', r'line\s+graph',
            r'trend', r'time\s+series'
        ],
        ChartType.SCATTER: [
            r'scatter\s+plot', r'scatter\s+chart', r'scatter\s+graph',
            r'correlation\s+plot'
        ],
        ChartType.REGRESSION: [
            r'regression\s+line', r'linear\s+regression', r'trend\s+line',
            r'best\s+fit\s+line', r'regression\s+plot', r'line\s+of\s+best\s+fit',
            r'trendline', r'fit\s+line', r'correlation\s+line', r'linear\s+trend',
            r'\bregression\b', r'best\s+fit', r'\btrend\b'
        ],
        ChartType.HISTOGRAM: [
            r'histogram', r'distribution', r'frequency\s+plot'
        ],
        ChartType.BOX: [
            r'box\s+plot', r'box\s+chart', r'boxplot'
        ],
        ChartType.PIE: [
            r'pie\s+chart', r'pie\s+plot', r'pie\s+graph'
        ],
        ChartType.HEATMAP: [
            r'heatmap', r'heat\s+map', r'correlation\s+matrix'
        ],
        ChartType.AREA: [
            r'area\s+chart', r'area\s+plot', r'filled\s+line'
        ]
    }
    
    @classmethod
    def contains_visualization_request(cls, query: str) -> bool:
        """
        Check if a query contains visualization keywords.
        
        Args:
            query: Natural language query
            
        Returns:
            True if query likely requests visualization
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in cls.VISUALIZATION_KEYWORDS)
    
    @classmethod
    def parse_visualization_request(cls, query: str, df_columns: List[str]) -> Optional[VisualizationRequest]:
        """
        Parse a natural language query to extract visualization parameters.
        
        Args:
            query: Natural language query
            df_columns: Available DataFrame columns
            
        Returns:
            VisualizationRequest object or None if no visualization detected
        """
        if not cls.contains_visualization_request(query):
            return None
        
        query_lower = query.lower()
        
        # Detect chart type
        chart_type = cls._detect_chart_type(query_lower)
        
        # Extract column references
        x_column, y_column = cls._extract_columns(query_lower, df_columns)
        
        # Extract color column if mentioned
        color_column = cls._extract_color_column(query_lower, df_columns)
        
        # Generate title
        title = cls._generate_title(query, chart_type, x_column, y_column)
        
        return VisualizationRequest(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title,
            additional_params={}
        )
    
    @classmethod
    def _detect_chart_type(cls, query_lower: str) -> ChartType:
        """Detect chart type from query text"""
        
        # First, check for explicit chart type patterns (highest priority)
        for chart_type, patterns in cls.CHART_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return chart_type
        
        # INTELLIGENT REGRESSION DETECTION - only if no explicit chart type found
        
        # Direct regression terms
        direct_regression = ['regression', 'linear regression', 'regression line', 'regression plot']
        
        # Trend and fit terms
        trend_terms = ['trend line', 'trendline', 'best fit', 'fit line', 'line of best fit']
        
        # Natural language patterns that imply regression
        natural_patterns = [
            'make a line', 'draw a line', 'add a line', 'show the line',
            'line based on', 'line through', 'line connecting'
        ]
        
        # Relationship terms that often need regression
        relationship_terms = ['linear relationship', 'correlation', 'relationship between']
        
        # Check for distribution first (highest priority among fallbacks)
        if any(word in query_lower for word in ['distribution', 'frequency']):
            return ChartType.HISTOGRAM
        
        # Check for regression indicators (second highest priority among fallbacks)
        elif any(term in query_lower for term in direct_regression):
            return ChartType.REGRESSION
        elif any(term in query_lower for term in trend_terms):
            return ChartType.REGRESSION
        elif any(pattern in query_lower for pattern in natural_patterns):
            return ChartType.REGRESSION
        elif any(term in query_lower for term in relationship_terms):
            return ChartType.REGRESSION
        
        # Additional regression detection for common phrases
        elif 'line' in query_lower and any(word in query_lower for word in ['between', 'vs', 'versus', 'and']):
            return ChartType.REGRESSION
        elif 'trend' in query_lower:
            return ChartType.REGRESSION
        elif 'fit' in query_lower:
            return ChartType.REGRESSION
        
        # Time series
        elif any(word in query_lower for word in ['over time', 'timeline']):
            return ChartType.LINE
        
        # Default
        else:
            return ChartType.BAR
    
    @classmethod
    def _extract_columns(cls, query_lower: str, df_columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extract column names from query text with improved matching"""
        mentioned_columns = []
        
        # Special handling for heatmap queries with "by" pattern (e.g., "profit by category and region")
        if 'heatmap' in query_lower and ' by ' in query_lower:
            # Extract the pattern: "value by column1 and column2"
            by_parts = query_lower.split(' by ')
            if len(by_parts) >= 2:
                # Look for columns mentioned after "by"
                after_by = by_parts[1]
                # Split on "and" to get multiple grouping columns
                grouping_parts = after_by.replace(' and ', '|').replace(',', '|').split('|')
                
                for part in grouping_parts:
                    part = part.strip()
                    # Find matching columns
                    for col in df_columns:
                        col_lower = col.lower()
                        col_with_spaces = col_lower.replace('_', ' ')
                        
                        if (col_lower in part or part in col_lower or 
                            col_with_spaces in part or part in col_with_spaces):
                            if col not in mentioned_columns:
                                mentioned_columns.append(col)
        
        # Regular column extraction if no special pattern found
        if not mentioned_columns:
            # Look for exact column matches (case insensitive)
            for col in df_columns:
                col_lower = col.lower()
                
                # Direct exact match
                if col_lower in query_lower:
                    mentioned_columns.append(col)
                    continue
                
                # Handle underscores - check if column name with spaces matches
                col_with_spaces = col_lower.replace('_', ' ')
                if col_with_spaces in query_lower:
                    mentioned_columns.append(col)
                    continue
                
                # Handle partial matches for compound column names
                col_parts = col_lower.replace('_', ' ').split()
                if len(col_parts) > 1:
                    # Check if all parts of the column name are mentioned
                    if all(part in query_lower for part in col_parts):
                        mentioned_columns.append(col)
                        continue
                
                # Check for column name mentioned after common keywords
                keywords = ['for', 'of', 'plot', 'chart', 'show', 'display', 'visualize']
                for keyword in keywords:
                    # Look for patterns like "box plot for Monthly_Allowance" or "chart of age"
                    pattern_exact = f"{keyword} {col_lower}"
                    pattern_spaces = f"{keyword} {col_with_spaces}"
                    if pattern_exact in query_lower or pattern_spaces in query_lower:
                        mentioned_columns.append(col)
                        break
        
        # Remove duplicates while preserving order
        unique_columns = []
        for col in mentioned_columns:
            if col not in unique_columns:
                unique_columns.append(col)
        
        # Return first two columns found, or None
        x_column = unique_columns[0] if len(unique_columns) > 0 else None
        y_column = unique_columns[1] if len(unique_columns) > 1 else None
        
        return x_column, y_column
    
    @classmethod
    def _extract_color_column(cls, query_lower: str, df_columns: List[str]) -> Optional[str]:
        """Extract color column from query text - for heatmaps, this represents the value column"""
        
        # Special handling for heatmap value extraction
        if 'heatmap' in query_lower:
            # Look for value column mentioned before "by" (e.g., "profit by category and region")
            if ' by ' in query_lower:
                before_by = query_lower.split(' by ')[0]
                # Look for value column names in the part before "by"
                value_candidates = ['profit', 'sales', 'revenue', 'amount', 'value', 'total', 'sum', 'count']
                
                for col in df_columns:
                    col_lower = col.lower()
                    col_with_spaces = col_lower.replace('_', ' ')
                    
                    # Check if column name appears before "by"
                    if col_lower in before_by or col_with_spaces in before_by:
                        return col
                    
                    # Check if column contains value-related terms
                    if any(candidate in col_lower for candidate in value_candidates):
                        if any(candidate in before_by for candidate in value_candidates):
                            return col
        
        # Regular color column extraction for other chart types
        # Look for color-related keywords
        color_keywords = ['color', 'colour', 'colored', 'coloured', 'by', 'group by', 'categorize', 'category']
        
        # Check if query mentions colors
        has_color_request = any(keyword in query_lower for keyword in color_keywords)
        
        if has_color_request:
            # Look for column names mentioned after color keywords
            for col in df_columns:
                col_lower = col.lower()
                # Check if column is mentioned in context of coloring
                for keyword in color_keywords:
                    if f"{keyword} {col_lower}" in query_lower or f"{keyword} by {col_lower}" in query_lower:
                        return col
                
                # Also check for direct mentions of categorical columns
                if col_lower in query_lower:
                    return col
        
        # Look for common categorical column patterns
        categorical_patterns = ['status', 'type', 'category', 'class', 'group', 'placed', 'result']
        for col in df_columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in categorical_patterns):
                return col
        
        return None
    
    @classmethod
    def _generate_title(cls, query: str, chart_type: ChartType, x_col: Optional[str], y_col: Optional[str]) -> str:
        """Generate a title for the visualization"""
        if x_col and y_col:
            return f"{chart_type.value.title()} Chart: {y_col} vs {x_col}"
        elif x_col:
            return f"{chart_type.value.title()} Chart: {x_col}"
        else:
            return f"{chart_type.value.title()} Chart"


class ChartGenerator:
    """Generator for creating charts using matplotlib and plotly"""
    
    def __init__(self, default_library: str = "plotly"):
        """
        Initialize chart generator.
        
        Args:
            default_library: Default library to use ("matplotlib" or "plotly")
        """
        self.default_library = default_library
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def generate_chart(self, df: pd.DataFrame, request: VisualizationRequest, 
                      library: Optional[str] = None) -> VisualizationResult:
        """
        Generate a chart based on the visualization request.
        
        Args:
            df: DataFrame to visualize
            request: VisualizationRequest object
            library: Library to use ("matplotlib" or "plotly")
            
        Returns:
            VisualizationResult object
        """
        library = library or self.default_library
        
        try:
            if library == "plotly":
                return self._generate_plotly_chart(df, request)
            else:
                return self._generate_matplotlib_chart(df, request)
        except Exception as e:
            # Use enhanced error handling if available
            if error_handler:
                error_info = error_handler.handle_error(e, f"Chart generation - {request.chart_type.value}")
                error_message = error_info.user_message
            else:
                error_message = f"Error creating {request.chart_type.value} chart: {str(e)}"
            
            return VisualizationResult(
                chart_data=None,
                chart_type=request.chart_type.value,
                library=library,
                title=request.title or "Chart",
                success=False,
                error_message=error_message
            )
    
    def _generate_plotly_chart(self, df: pd.DataFrame, request: VisualizationRequest) -> VisualizationResult:
        """Generate chart using Plotly"""
        chart_type = request.chart_type
        
        if chart_type == ChartType.BAR:
            fig = self._create_plotly_bar(df, request)
        elif chart_type == ChartType.LINE:
            fig = self._create_plotly_line(df, request)
        elif chart_type == ChartType.SCATTER:
            fig = self._create_plotly_scatter(df, request)
        elif chart_type == ChartType.HISTOGRAM:
            fig = self._create_plotly_histogram(df, request)
        elif chart_type == ChartType.BOX:
            fig = self._create_plotly_box(df, request)
        elif chart_type == ChartType.PIE:
            fig = self._create_plotly_pie(df, request)
        elif chart_type == ChartType.HEATMAP:
            fig = self._create_plotly_heatmap(df, request)
        elif chart_type == ChartType.AREA:
            fig = self._create_plotly_area(df, request)
        elif chart_type == ChartType.REGRESSION:
            fig = self._create_plotly_regression(df, request)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return VisualizationResult(
            chart_data=fig,
            chart_type=chart_type.value,
            library="plotly",
            title=request.title or "Chart"
        )
    
    def _create_plotly_bar(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly bar chart with enhanced color mapping"""
        if request.x_column and request.y_column:
            # Check if we have a color column specified
            color_col = request.color_column if request.color_column else None
            
            # If no color column specified, try to use the x_column for color if it's categorical
            if not color_col and request.x_column:
                if df[request.x_column].dtype == 'object' or df[request.x_column].nunique() <= 10:
                    color_col = request.x_column
            
            # Create bar chart with color mapping
            if color_col:
                # Define custom color mapping for common binary categories
                color_map = self._get_color_mapping(df, color_col)
                fig = px.bar(df, x=request.x_column, y=request.y_column, 
                           color=color_col, title=request.title, color_discrete_map=color_map)
            else:
                fig = px.bar(df, x=request.x_column, y=request.y_column, title=request.title)
                
        elif request.x_column:
            # Count values in x_column
            value_counts = df[request.x_column].value_counts().head(20)
            
            # Create color mapping for value counts
            colors = self._get_colors_for_categories(value_counts.index.tolist())
            
            # Create DataFrame for proper color mapping
            chart_df = pd.DataFrame({
                'category': value_counts.index,
                'count': value_counts.values
            })
            
            fig = px.bar(chart_df, x='category', y='count', 
                        title=request.title, color='category',
                        color_discrete_sequence=colors)
            fig.update_xaxis(title=request.x_column)
            fig.update_yaxis(title="Count")
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                value_counts = df[col].value_counts().head(20)
                colors = self._get_colors_for_categories(value_counts.index.tolist())
                
                # Create DataFrame for proper color mapping
                chart_df = pd.DataFrame({
                    'category': value_counts.index,
                    'count': value_counts.values
                })
                
                fig = px.bar(chart_df, x='category', y='count', 
                           title=request.title, color='category',
                           color_discrete_sequence=colors)
                fig.update_xaxis(title=col)
                fig.update_yaxis(title="Count")
            else:
                raise ValueError("No suitable columns found for bar chart")
        
        return fig
    
    def _get_color_mapping(self, df: pd.DataFrame, color_col: str) -> Dict[str, str]:
        """Get appropriate color mapping for categorical data"""
        unique_values = df[color_col].unique()
        
        # Special handling for common binary categories
        if len(unique_values) == 2:
            values_lower = [str(v).lower() for v in unique_values]
            
            # Placement status (placed/not placed)
            if any(word in ' '.join(values_lower) for word in ['placed', 'not placed', 'yes', 'no']):
                placed_indicators = ['placed', 'yes', '1', 'true', 'success']
                not_placed_indicators = ['not placed', 'no', '0', 'false', 'fail']
                
                color_map = {}
                for val in unique_values:
                    val_lower = str(val).lower()
                    if any(indicator in val_lower for indicator in placed_indicators):
                        color_map[val] = '#2E8B57'  # Green for placed/success
                    elif any(indicator in val_lower for indicator in not_placed_indicators):
                        color_map[val] = '#DC143C'  # Red for not placed/failure
                    else:
                        # Fallback: first value green, second red
                        color_map[unique_values[0]] = '#2E8B57'
                        color_map[unique_values[1]] = '#DC143C'
                
                return color_map
        
        # Default color mapping for multiple categories
        return {}
    
    def _get_colors_for_categories(self, categories: List[str]) -> List[str]:
        """Get appropriate colors for a list of categories"""
        # Default color palette
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Special handling for binary categories
        if len(categories) == 2:
            categories_lower = [str(cat).lower() for cat in categories]
            
            # Check for placement/success indicators
            placed_indicators = ['placed', 'yes', '1', 'true', 'success']
            not_placed_indicators = ['not placed', 'no', '0', 'false', 'fail']
            
            colors = []
            for cat in categories:
                cat_lower = str(cat).lower()
                if any(indicator in cat_lower for indicator in placed_indicators):
                    colors.append('#2E8B57')  # Green
                elif any(indicator in cat_lower for indicator in not_placed_indicators):
                    colors.append('#DC143C')  # Red
                else:
                    colors.append(default_colors[len(colors) % len(default_colors)])
            
            return colors
        
        # Return default colors for multiple categories
        return default_colors[:len(categories)]
    
    def _create_plotly_line(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly line chart"""
        if request.x_column and request.y_column:
            fig = px.line(df, x=request.x_column, y=request.y_column, title=request.title)
        else:
            # Use index and first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.line(df, y=numeric_cols[0], title=request.title)
            else:
                raise ValueError("No suitable columns found for line chart")
        
        return fig
    
    def _create_plotly_scatter(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly scatter plot"""
        if request.x_column and request.y_column:
            fig = px.scatter(df, x=request.x_column, y=request.y_column, title=request.title)
        else:
            # Use first two numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=request.title)
            else:
                raise ValueError("Need at least two numeric columns for scatter plot")
        
        return fig
    
    def _create_plotly_histogram(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly histogram"""
        if request.x_column:
            fig = px.histogram(df, x=request.x_column, title=request.title)
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], title=request.title)
            else:
                raise ValueError("No suitable columns found for histogram")
        
        return fig
    
    def _create_plotly_box(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly box plot"""
        # For box plots, use y_column first, then x_column, then fallback to first numeric
        target_column = None
        
        if request.y_column:
            target_column = request.y_column
        elif request.x_column:
            target_column = request.x_column
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[0]
            else:
                raise ValueError("No suitable columns found for box plot")
        
        # Verify the column exists and is numeric
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Column '{target_column}' is not numeric and cannot be used for box plot")
        
        fig = px.box(df, y=target_column, title=request.title or f"Box Plot: {target_column}")
        
        return fig
    
    def _create_plotly_pie(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly pie chart"""
        if request.x_column:
            value_counts = df[request.x_column].value_counts().head(10)
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=request.title)
        else:
            # Use first categorical column
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                value_counts = df[col].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=request.title)
            else:
                raise ValueError("No suitable columns found for pie chart")
        
        return fig
    
    def _create_plotly_heatmap(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly heatmap - supports both correlation matrices and pivot table heatmaps"""
        
        # Check if we have specific columns for pivot table heatmap
        if request.x_column and request.y_column:
            # Create pivot table heatmap
            # x_column = categories (e.g., Category), y_column = grouping (e.g., Region)
            # We need a value column - use color_column if specified, otherwise find a numeric column
            
            value_column = None
            if request.color_column and request.color_column in df.columns:
                if pd.api.types.is_numeric_dtype(df[request.color_column]):
                    value_column = request.color_column
            
            # If no value column specified, try to infer from common patterns
            if value_column is None:
                # Look for common value column names
                value_candidates = ['profit', 'sales', 'revenue', 'amount', 'value', 'total']
                for col in df.columns:
                    if any(candidate in col.lower() for candidate in value_candidates):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            value_column = col
                            break
                
                # If still no value column, use the first numeric column
                if value_column is None:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        value_column = numeric_cols[0]
                    else:
                        raise ValueError("No numeric column found for heatmap values")
            
            # Create pivot table
            try:
                pivot_table = df.pivot_table(
                    values=value_column,
                    index=request.y_column,  # Rows (e.g., Region)
                    columns=request.x_column,  # Columns (e.g., Category)
                    aggfunc='sum',  # Aggregate function
                    fill_value=0
                )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    text_auto=True,
                    aspect="auto",
                    title=request.title or f"{value_column} by {request.x_column} and {request.y_column}",
                    labels={'color': value_column}
                )
                
                # Update layout for better readability
                fig.update_layout(
                    xaxis_title=request.x_column,
                    yaxis_title=request.y_column
                )
                
                return fig
                
            except Exception as e:
                raise ValueError(f"Could not create pivot table heatmap: {str(e)}")
        
        else:
            # Default to correlation matrix heatmap
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                raise ValueError("Need at least two numeric columns for correlation heatmap")
            
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto", 
                title=request.title or "Correlation Matrix"
            )
            
            return fig
    
    def _create_plotly_area(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly area chart"""
        if request.x_column and request.y_column:
            fig = px.area(df, x=request.x_column, y=request.y_column, title=request.title)
        else:
            # Use index and first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.area(df, y=numeric_cols[0], title=request.title)
            else:
                raise ValueError("No suitable columns found for area chart")
        
        return fig
    
    def _create_plotly_regression(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly scatter plot with visible regression line"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        if request.x_column and request.y_column:
            x_col, y_col = request.x_column, request.y_column
        else:
            # Use first two numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                raise ValueError("Need at least two numeric columns for regression plot")
        
        # Get clean data (remove NaN values)
        clean_df = df[[x_col, y_col]].dropna()
        x_data = clean_df[x_col].values
        y_data = clean_df[y_col].values
        
        # Create scatter plot
        if request.color_column:
            color_map = self._get_color_mapping(df, request.color_column)
            fig = px.scatter(df, x=x_col, y=y_col, color=request.color_column,
                           title=request.title, color_discrete_map=color_map)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=request.title)
        
        # Calculate regression line manually for better visibility
        if len(x_data) > 1:
            # Fit linear regression
            X = x_data.reshape(-1, 1)
            reg = LinearRegression().fit(X, y_data)
            
            # Generate line points
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))
            
            # Add regression line as a separate trace
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'Regression Line (R² = {reg.score(X, y_data):.3f})',
                line=dict(color='red', width=3, dash='solid'),
                showlegend=True
            ))
            
            # Add regression equation as annotation
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r_squared = reg.score(X, y_data)
            
            equation_text = f"y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_squared:.3f}"
            
            fig.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=equation_text,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        return fig
    
    def _generate_matplotlib_chart(self, df: pd.DataFrame, request: VisualizationRequest) -> VisualizationResult:
        """Generate chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_type = request.chart_type
        
        try:
            if chart_type == ChartType.BAR:
                self._create_matplotlib_bar(df, request, ax)
            elif chart_type == ChartType.LINE:
                self._create_matplotlib_line(df, request, ax)
            elif chart_type == ChartType.SCATTER:
                self._create_matplotlib_scatter(df, request, ax)
            elif chart_type == ChartType.HISTOGRAM:
                self._create_matplotlib_histogram(df, request, ax)
            elif chart_type == ChartType.BOX:
                self._create_matplotlib_box(df, request, ax)
            else:
                raise ValueError(f"Matplotlib chart type {chart_type} not implemented")
            
            ax.set_title(request.title or "Chart")
            plt.tight_layout()
            
            return VisualizationResult(
                chart_data=fig,
                chart_type=chart_type.value,
                library="matplotlib",
                title=request.title or "Chart"
            )
            
        except Exception as e:
            plt.close(fig)
            raise e
    
    def _create_matplotlib_bar(self, df: pd.DataFrame, request: VisualizationRequest, ax):
        """Create matplotlib bar chart"""
        if request.x_column and request.y_column:
            df.plot(x=request.x_column, y=request.y_column, kind='bar', ax=ax)
        elif request.x_column:
            value_counts = df[request.x_column].value_counts().head(20)
            value_counts.plot(kind='bar', ax=ax)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].value_counts().head(20).plot(kind='bar', ax=ax)
            else:
                raise ValueError("No suitable columns found for bar chart")
    
    def _create_matplotlib_line(self, df: pd.DataFrame, request: VisualizationRequest, ax):
        """Create matplotlib line chart"""
        if request.x_column and request.y_column:
            df.plot(x=request.x_column, y=request.y_column, kind='line', ax=ax)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].plot(kind='line', ax=ax)
            else:
                raise ValueError("No suitable columns found for line chart")
    
    def _create_matplotlib_scatter(self, df: pd.DataFrame, request: VisualizationRequest, ax):
        """Create matplotlib scatter plot"""
        if request.x_column and request.y_column:
            df.plot(x=request.x_column, y=request.y_column, kind='scatter', ax=ax)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                df.plot(x=numeric_cols[0], y=numeric_cols[1], kind='scatter', ax=ax)
            else:
                raise ValueError("Need at least two numeric columns for scatter plot")
    
    def _create_matplotlib_histogram(self, df: pd.DataFrame, request: VisualizationRequest, ax):
        """Create matplotlib histogram"""
        if request.x_column:
            df[request.x_column].hist(ax=ax, bins=30)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].hist(ax=ax, bins=30)
            else:
                raise ValueError("No suitable columns found for histogram")
    
    def _create_matplotlib_box(self, df: pd.DataFrame, request: VisualizationRequest, ax):
        """Create matplotlib box plot"""
        # For box plots, use y_column first, then x_column, then fallback to first numeric
        target_column = None
        
        if request.y_column:
            target_column = request.y_column
        elif request.x_column:
            target_column = request.x_column
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[0]
            else:
                raise ValueError("No suitable columns found for box plot")
        
        # Verify the column exists and is numeric
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Column '{target_column}' is not numeric and cannot be used for box plot")
        
        df.boxplot(column=target_column, ax=ax)
        ax.set_title(request.title or f"Box Plot: {target_column}")


class VisualizationTool:
    """
    Main visualization tool that integrates parsing and chart generation.
    This tool can be used by the LangChain agent for visualization tasks.
    """
    
    def __init__(self, default_library: str = "plotly"):
        """
        Initialize visualization tool.
        
        Args:
            default_library: Default library to use for charts
        """
        self.parser = VisualizationParser()
        self.generator = ChartGenerator(default_library)
    
    def create_visualization(self, query: str, df: pd.DataFrame, 
                           library: Optional[str] = None) -> Optional[VisualizationResult]:
        """
        Create visualization from natural language query and DataFrame.
        
        Args:
            query: Natural language query
            df: DataFrame to visualize
            library: Optional library preference
            
        Returns:
            VisualizationResult or None if no visualization requested
        """
        # Parse the query for visualization request
        viz_request = self.parser.parse_visualization_request(query, df.columns.tolist())
        
        if not viz_request:
            return None
        
        # Generate the chart
        return self.generator.generate_chart(df, viz_request, library)
    
    def is_visualization_query(self, query: str) -> bool:
        """Check if query requests visualization"""
        return self.parser.contains_visualization_request(query)


class VisualizationManager:
    """
    Manager class for handling visualization requests and chart generation.
    Provides a simplified interface for creating charts from DataFrames.
    """
    
    def __init__(self, default_library: str = "plotly"):
        """
        Initialize visualization manager.
        
        Args:
            default_library: Default library to use for charts
        """
        self.generator = ChartGenerator(default_library)
        self.default_library = default_library
    
    def create_chart(self, df: pd.DataFrame, chart_type: ChartType, 
                    x: Optional[str] = None, y: Optional[str] = None,
                    title: Optional[str] = None, **kwargs) -> VisualizationResult:
        """
        Create a chart with specified parameters.
        
        Args:
            df: DataFrame to visualize
            chart_type: Type of chart to create
            x: X-axis column name
            y: Y-axis column name
            title: Chart title
            **kwargs: Additional parameters
            
        Returns:
            VisualizationResult object
        """
        # Create visualization request
        request = VisualizationRequest(
            chart_type=chart_type,
            x_column=x,
            y_column=y,
            title=title or f"{chart_type.value.title()} Chart",
            additional_params=kwargs
        )
        
        # Generate the chart
        return self.generator.generate_chart(df, request)
    
    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, 
                        title: Optional[str] = None) -> VisualizationResult:
        """Create a bar chart"""
        return self.create_chart(df, ChartType.BAR, x=x, y=y, title=title)
    
    def create_line_chart(self, df: pd.DataFrame, x: str, y: str,
                         title: Optional[str] = None) -> VisualizationResult:
        """Create a line chart"""
        return self.create_chart(df, ChartType.LINE, x=x, y=y, title=title)
    
    def create_scatter_plot(self, df: pd.DataFrame, x: str, y: str,
                           title: Optional[str] = None) -> VisualizationResult:
        """Create a scatter plot"""
        return self.create_chart(df, ChartType.SCATTER, x=x, y=y, title=title)
    
    def create_histogram(self, df: pd.DataFrame, column: str,
                        title: Optional[str] = None) -> VisualizationResult:
        """Create a histogram"""
        return self.create_chart(df, ChartType.HISTOGRAM, x=column, title=title)
    
    def create_box_plot(self, df: pd.DataFrame, column: str,
                       title: Optional[str] = None) -> VisualizationResult:
        """Create a box plot"""
        return self.create_chart(df, ChartType.BOX, y=column, title=title)
    
    def create_pie_chart(self, df: pd.DataFrame, column: str,
                        title: Optional[str] = None) -> VisualizationResult:
        """Create a pie chart"""
        return self.create_chart(df, ChartType.PIE, x=column, title=title)
    
    def create_regression_plot(self, df: pd.DataFrame, x: str, y: str,
                              title: Optional[str] = None) -> VisualizationResult:
        """Create a scatter plot with regression line"""
        return self.create_chart(df, ChartType.REGRESSION, x=x, y=y, title=title)
    
    def get_supported_chart_types(self) -> List[ChartType]:
        """Get list of supported chart types"""
        return list(ChartType)
    
    def suggest_chart_type(self, df: pd.DataFrame, x_col: Optional[str] = None, 
                          y_col: Optional[str] = None) -> ChartType:
        """
        Suggest appropriate chart type based on data characteristics.
        
        Args:
            df: DataFrame to analyze
            x_col: X-axis column name
            y_col: Y-axis column name
            
        Returns:
            Suggested ChartType
        """
        if x_col and y_col:
            x_dtype = df[x_col].dtype
            y_dtype = df[y_col].dtype
            
            # Both numeric -> scatter plot
            if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
                return ChartType.SCATTER
            
            # X categorical, Y numeric -> bar chart
            elif not pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
                return ChartType.BAR
            
            # X numeric, Y categorical -> horizontal bar (use bar for now)
            elif pd.api.types.is_numeric_dtype(x_dtype) and not pd.api.types.is_numeric_dtype(y_dtype):
                return ChartType.BAR
            
            # Both categorical -> bar chart of counts
            else:
                return ChartType.BAR
        
        elif x_col:
            x_dtype = df[x_col].dtype
            
            # Single numeric column -> histogram
            if pd.api.types.is_numeric_dtype(x_dtype):
                return ChartType.HISTOGRAM
            
            # Single categorical column -> bar chart of counts
            else:
                return ChartType.BAR
        
        else:
            # No specific columns -> use first numeric column for histogram
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return ChartType.HISTOGRAM
            else:
                return ChartType.BAR
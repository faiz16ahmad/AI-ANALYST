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
    TREEMAP = "treemap"  # Treemap visualization
    
    # Complex chart types (handled by agent code generation)
    DUAL_AXIS = "dual_axis"
    COMBO = "combo"
    COMPLEX_MULTI = "complex_multi"
    CUSTOM = "custom"


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
        'pie chart', 'box plot', 'heatmap', 'treemap', 'distribution',
        'compare', 'comparison', 'trend', 'analyze', 'analysis',
        'relationship', 'correlation', 'breakdown', 'across'
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
        ChartType.TREEMAP: [
            r'treemap', r'tree\s+map', r'hierarchical\s+chart'
        ],
        ChartType.AREA: [
            r'area\s+chart', r'area\s+plot', r'filled\s+line'
        ],
        ChartType.DUAL_AXIS: [
            r'dual\s+axis', r'dual-axis', r'two\s+y\s+axis', r'secondary\s+axis',
            r'left.*right.*axis', r'bars.*line', r'combo.*chart'
        ],
        ChartType.COMBO: [
            r'combo\s+chart', r'combination\s+chart', r'mixed\s+chart'
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
        Parse a natural language query to extract visualization parameters using intelligent analysis.
        
        Args:
            query: Natural language query
            df_columns: Available DataFrame columns
            
        Returns:
            VisualizationRequest object or None if no visualization detected
        """
        if not cls.contains_visualization_request(query):
            return None
        
        query_lower = query.lower()
        
        # Extract column references first (helps with chart type detection)
        x_column, y_column = cls._extract_columns(query_lower, df_columns)
        color_column = cls._extract_color_column(query_lower, df_columns)
        
        # Use intelligent chart type detection with column context
        chart_type = cls._detect_chart_type_with_context(query_lower, x_column, y_column, color_column, df_columns)
        
        # Assess confidence level - CRITICAL for deciding tool vs LLM
        confidence_score = cls._assess_confidence(query_lower, chart_type, x_column, y_column, color_column, df_columns)
        
        # Generate title
        title = cls._generate_title(query, chart_type, x_column, y_column)
        
        return VisualizationRequest(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title,
            additional_params={'confidence_score': confidence_score}
        )
    
    @classmethod
    def _assess_confidence_llm_based(cls, query: str, df_columns: List[str]) -> float:
        """
        Alternative: Use LLM to assess confidence (more accurate but slower)
        """
        # This would be implemented if we want LLM-based assessment
        # For now, we'll stick with rule-based for speed
        pass
    
    @classmethod
    def _assess_confidence(cls, query_lower: str, chart_type: ChartType, 
                          x_column: Optional[str], y_column: Optional[str], 
                          color_column: Optional[str], df_columns: List[str]) -> float:
        """
        Assess confidence level for using our tool vs LLM code generation.
        
        Returns:
            Float between 0.0 (no confidence, use LLM) and 1.0 (high confidence, use tool)
        """
        confidence = 0.0
        
        # HIGH CONFIDENCE INDICATORS (use our tool)
        
        # 1. Explicit chart type mentioned
        explicit_chart_patterns = [
            r'\bbar chart\b', r'\bline chart\b', r'\bscatter plot\b', 
            r'\bhistogram\b', r'\bbox plot\b', r'\bpie chart\b', r'\bheatmap\b'
        ]
        if any(re.search(pattern, query_lower) for pattern in explicit_chart_patterns):
            confidence += 0.4
        
        # 2. Simple, clear structure
        simple_patterns = [
            r'show.*chart', r'create.*plot', r'generate.*graph',
            r'plot.*vs', r'chart.*by', r'histogram.*of'
        ]
        if any(re.search(pattern, query_lower) for pattern in simple_patterns):
            confidence += 0.3
        
        # 3. Clear column references
        if x_column and x_column in df_columns:
            confidence += 0.2
        if y_column and y_column in df_columns:
            confidence += 0.2
        
        # 4. Standard chart types we handle well
        high_confidence_types = [ChartType.BAR, ChartType.LINE, ChartType.SCATTER, 
                               ChartType.HISTOGRAM, ChartType.BOX, ChartType.PIE, ChartType.HEATMAP, ChartType.TREEMAP]
        if chart_type in high_confidence_types:
            confidence += 0.3
        
        # LOW CONFIDENCE INDICATORS (use LLM)
        
        # 1. Complex styling requirements
        complex_styling = [
            'gradient', 'custom color', 'data labels', 'annotations',
            'styling', 'format', 'theme', 'advanced', 'sophisticated'
        ]
        if any(term in query_lower for term in complex_styling):
            confidence -= 0.4
        
        # 2. Multiple chart types or complex combinations
        multi_chart_indicators = [
            'and', 'with', 'plus', 'combined', 'overlay', 'multiple'
        ]
        chart_type_count = sum(1 for pattern in explicit_chart_patterns 
                              if re.search(pattern, query_lower))
        if chart_type_count > 1 or any(term in query_lower for term in multi_chart_indicators):
            confidence -= 0.3
        
        # 3. Vague or ambiguous requests
        vague_terms = [
            'analyze', 'explore', 'investigate', 'understand', 'insights',
            'patterns', 'trends', 'relationships', 'overview'
        ]
        if any(term in query_lower for term in vague_terms) and not any(re.search(pattern, query_lower) for pattern in explicit_chart_patterns):
            confidence -= 0.3
        
        # 4. Specific technical requirements
        technical_requirements = [
            'axis', 'scale', 'log', 'normalize', 'aggregate', 'group by',
            'filter', 'sort', 'transform', 'calculate'
        ]
        if any(term in query_lower for term in technical_requirements):
            confidence -= 0.2
        
        # 5. Long, complex queries (>100 characters with multiple clauses)
        if len(query_lower) > 100 and query_lower.count(',') > 2:
            confidence -= 0.2
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    @classmethod
    def _detect_chart_type_with_context(cls, query_lower: str, x_column: Optional[str], 
                                       y_column: Optional[str], color_column: Optional[str],
                                       df_columns: List[str]) -> ChartType:
        """
        Detect chart type using both query text and column context for better accuracy.
        """
        # First, check for explicit chart type mentions
        for chart_type, patterns in cls.CHART_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return chart_type
        
        # Use context-aware intelligent detection
        return cls._context_aware_chart_detection(query_lower, x_column, y_column, color_column, df_columns)
    
    @classmethod
    def _context_aware_chart_detection(cls, query_lower: str, x_column: Optional[str], 
                                     y_column: Optional[str], color_column: Optional[str],
                                     df_columns: List[str]) -> ChartType:
        """
        Determine chart type based on query intent AND data structure context.
        """
        # Analyze the data structure context
        has_two_dimensions = x_column and y_column
        has_value_column = color_column
        
        # Special case: Multi-dimensional aggregation queries (heatmap territory)
        aggregation_patterns = ['by', 'across', 'grouped by', 'broken down by']
        multi_dim_indicators = [' and ', ',', 'vs', 'versus']
        
        if any(pattern in query_lower for pattern in aggregation_patterns):
            # Check if query mentions multiple grouping dimensions
            for pattern in aggregation_patterns:
                if pattern in query_lower:
                    after_pattern = query_lower.split(pattern, 1)[1]
                    if any(indicator in after_pattern for indicator in multi_dim_indicators):
                        # Only heatmap if we actually have two dimensions AND a value
                        if has_two_dimensions and (has_value_column or 'profit' in query_lower or 'sales' in query_lower):
                            return ChartType.HEATMAP
        
        # Intent-based detection with data context
        if 'distribution' in query_lower or 'frequency' in query_lower:
            return ChartType.HISTOGRAM
        
        elif any(word in query_lower for word in ['correlation', 'relationship', 'association']):
            if 'matrix' in query_lower or not has_two_dimensions:
                return ChartType.HEATMAP  # Correlation matrix
            elif any(word in query_lower for word in ['linear', 'regression', 'trend line', 'fit']):
                return ChartType.REGRESSION
            else:
                return ChartType.SCATTER
        
        elif any(word in query_lower for word in ['trend', 'over time', 'timeline', 'change']):
            return ChartType.LINE
        
        elif any(word in query_lower for word in ['proportion', 'percentage', 'share', 'part of']):
            return ChartType.PIE
        
        elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
            if has_two_dimensions and has_value_column:
                return ChartType.HEATMAP  # Comparison across two dimensions
            else:
                return ChartType.BAR
        
        # Check for dual-axis specific patterns first
        dual_axis_indicators = [
            'dual axis', 'dual-axis', 'secondary axis', 'left y-axis', 'right y-axis',
            'bars and line', 'bar and line', 'two y axis', 'two y-axis'
        ]
        
        if any(indicator in query_lower for indicator in dual_axis_indicators):
            return ChartType.DUAL_AXIS
        
        # Check for combo chart patterns (bars + line combination)
        has_bars = any(word in query_lower for word in ['bars for', 'use bars', 'bar chart'])
        has_line = any(word in query_lower for word in ['line for', 'use line', 'line chart'])
        
        if has_bars and has_line:
            return ChartType.DUAL_AXIS
        
        # Check for complex styling that requires fallback
        complex_styling = ['gradient color', 'data labels', 'custom styling', 'advanced formatting']
        if any(indicator in query_lower for indicator in complex_styling):
            return ChartType.COMPLEX_MULTI  # Still needs fallback for advanced styling
        
        # Special handling for "value by category" patterns (should be bar chart, not heatmap)
        if ' by ' in query_lower:
            by_parts = query_lower.split(' by ')
            if len(by_parts) == 2:  # Simple "X by Y" pattern
                after_by = by_parts[1].strip()
                # If there's no "and" or comma after "by", it's single dimension grouping
                if not any(indicator in after_by for indicator in multi_dim_indicators):
                    return ChartType.BAR
        
        # Default based on data structure and query patterns
        if has_two_dimensions and has_value_column:
            # Three variables detected -> likely heatmap
            return ChartType.HEATMAP
        elif has_two_dimensions:
            # Two variables -> scatter plot
            return ChartType.SCATTER
        else:
            # Default fallback
            return ChartType.BAR
    
    @classmethod
    def _detect_chart_type(cls, query_lower: str) -> ChartType:
        """Detect chart type from query text using intelligent analysis"""
        
        # First, check for explicit chart type mentions (highest confidence)
        for chart_type, patterns in cls.CHART_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return chart_type
        
        # If no explicit chart type, use intelligent intent detection
        return cls._intelligent_chart_detection(query_lower)
    
    @classmethod
    def _intelligent_chart_detection(cls, query_lower: str) -> ChartType:
        """Use intelligent analysis to determine the best chart type based on intent"""
        
        # Analyze query intent patterns
        intent_indicators = {
            'distribution': ['distribution', 'spread', 'frequency', 'how many', 'count of'],
            'comparison': ['compare', 'versus', 'vs', 'difference between', 'which is higher', 'top', 'bottom'],
            'relationship': ['relationship', 'correlation', 'association', 'connected', 'related'],
            'trend': ['trend', 'over time', 'timeline', 'change', 'growth', 'decline'],
            'composition': ['part of', 'percentage', 'proportion', 'share', 'breakdown'],
            'aggregation': ['total', 'sum', 'average', 'by category', 'by region', 'grouped by']
        }
        
        # Count intent matches
        intent_scores = {}
        for intent, indicators in intent_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Determine chart type based on strongest intent
        if not intent_scores:
            return ChartType.BAR  # Default fallback
        
        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        
        # Map intents to appropriate chart types
        if primary_intent == 'distribution':
            return ChartType.HISTOGRAM
        elif primary_intent == 'comparison':
            # Check if it's a heatmap-style comparison (by two dimensions)
            if ' by ' in query_lower and len(query_lower.split(' by ')) > 1:
                after_by = query_lower.split(' by ')[1]
                if ' and ' in after_by or ',' in after_by:
                    return ChartType.HEATMAP
            return ChartType.BAR
        elif primary_intent == 'relationship':
            # Check for regression indicators
            regression_terms = ['linear', 'regression', 'fit', 'predict']
            if any(term in query_lower for term in regression_terms):
                return ChartType.REGRESSION
            return ChartType.SCATTER
        elif primary_intent == 'trend':
            return ChartType.LINE
        elif primary_intent == 'composition':
            return ChartType.PIE
        elif primary_intent == 'aggregation':
            # Check if it's multi-dimensional aggregation (heatmap territory)
            if ' by ' in query_lower:
                after_by = query_lower.split(' by ')[1]
                if ' and ' in after_by or ',' in after_by:
                    return ChartType.HEATMAP
            return ChartType.BAR
        
        return ChartType.BAR
    
    @classmethod
    def _extract_columns(cls, query_lower: str, df_columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extract column names from query text with improved matching"""
        mentioned_columns = []
        
        # Special handling for dual-axis queries
        if any(indicator in query_lower for indicator in ['dual axis', 'dual-axis', 'bars and line', 'bar and line']):
            # For dual-axis, we need to extract the grouping column and two metrics
            # Look for patterns like "monthly sales and profit" or "sales by month"
            
            # Find all column mentions
            for col in df_columns:
                col_lower = col.lower()
                col_with_spaces = col_lower.replace('_', ' ')
                
                if col_lower in query_lower or col_with_spaces in query_lower:
                    if col not in mentioned_columns:
                        mentioned_columns.append(col)
        
        # Special handling for heatmap queries with "by" pattern (e.g., "profit by category and region")
        elif 'heatmap' in query_lower and ' by ' in query_lower:
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
        
        # Special handling for dual-axis charts
        if any(indicator in query_lower for indicator in ['dual axis', 'dual-axis', 'bars and line', 'bar and line']):
            # For dual-axis, color_column represents the second metric (line chart)
            
            # Look for patterns like "bars for X and line for Y"
            if 'line for' in query_lower:
                line_part = query_lower.split('line for')[1].strip()
                # Extract the first word after "line for"
                line_metric = line_part.split()[0] if line_part else ''
                for col in df_columns:
                    col_lower = col.lower()
                    if col_lower in line_metric or line_metric in col_lower:
                        return col
            
            # Look for "profit" specifically mentioned (common second metric)
            if 'profit' in query_lower:
                for col in df_columns:
                    if 'profit' in col.lower():
                        return col
            
            # Find all numeric columns mentioned and return the one that's different from y_column
            numeric_mentions = []
            for col in df_columns:
                col_lower = col.lower()
                if col_lower in query_lower:
                    # Check if it's a numeric-sounding column
                    if any(term in col_lower for term in ['profit', 'sales', 'revenue', 'amount', 'total', 'count']):
                        numeric_mentions.append(col)
            
            # Return a different column than what might be in y_column
            # This is a simple heuristic - in practice, the LLM should be more specific
            if len(numeric_mentions) >= 2:
                return numeric_mentions[-1]  # Return the last one found
        
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
        Generate a chart based on the visualization request with confidence-based routing.
        
        Args:
            df: DataFrame to visualize
            request: VisualizationRequest object
            library: Library to use ("matplotlib" or "plotly")
            
        Returns:
            VisualizationResult object
        """
        library = library or self.default_library
        
        # CONFIDENCE-BASED ROUTING: Only use our tool when we're confident
        confidence_score = request.additional_params.get('confidence_score', 0.0) if request.additional_params else 0.0
        
        # Dynamic threshold based on chart type complexity
        if request.chart_type in [ChartType.BAR, ChartType.LINE, ChartType.SCATTER]:
            confidence_threshold = 0.6  # Lower threshold for simple charts
        elif request.chart_type in [ChartType.HISTOGRAM, ChartType.BOX, ChartType.PIE, ChartType.TREEMAP]:
            confidence_threshold = 0.7  # Standard threshold
        else:
            confidence_threshold = 0.8  # Higher threshold for complex charts
        
        # SIMPLIFIED ROUTING: Only fallback for explicitly complex requests
        # Don't use confidence scores - they're unreliable
        
        # Check for explicit complexity indicators based on chart type and parameters
        explicit_complex = (
            request.chart_type in [ChartType.DUAL_AXIS, ChartType.COMPLEX_MULTI] or
            (request.additional_params and 
             any(key in request.additional_params for key in ['gradient', 'custom_color', 'data_labels', 'annotations']))
        )
        
        if explicit_complex:
            # Explicitly complex - let LLM handle it
            return VisualizationResult(
                chart_data=None,
                chart_type=request.chart_type.value,
                library=library,
                title=request.title or "Chart",
                success=False,
                error_message=f"FALLBACK_TO_CODE:explicit_complexity"
            )
        
        try:
            if library == "plotly":
                return self._generate_plotly_chart(df, request)
            else:
                return self._generate_matplotlib_chart(df, request)
        except Exception as e:
            error_str = str(e)
            
            # Check if this is a fallback request (not a real error)
            if "FALLBACK_TO_CODE:" in error_str:
                return VisualizationResult(
                    chart_data=None,
                    chart_type=request.chart_type.value,
                    library=library,
                    title=request.title or "Chart",
                    success=False,
                    error_message=error_str  # Pass through the fallback signal
                )
            
            # Use enhanced error handling for real errors
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
        elif chart_type == ChartType.TREEMAP:
            fig = self._create_plotly_treemap(df, request)
        elif chart_type == ChartType.AREA:
            fig = self._create_plotly_area(df, request)
        elif chart_type == ChartType.REGRESSION:
            fig = self._create_plotly_regression(df, request)
        elif chart_type == ChartType.DUAL_AXIS:
            fig = self._create_plotly_dual_axis(df, request)
        elif chart_type in [ChartType.COMBO, ChartType.COMPLEX_MULTI, ChartType.CUSTOM]:
            # These chart types still require agent code generation
            raise ValueError(f"FALLBACK_TO_CODE:{chart_type.value}")
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
    
    def _create_plotly_treemap(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly treemap visualization"""
        import plotly.express as px
        
        # For treemaps, we need a categorical column and a value column
        # x_column = categorical (path), color_column or y_column = values
        
        if request.x_column:
            # Use x_column as the categorical path
            path_column = request.x_column
            
            # Determine value column
            value_column = None
            if request.color_column and pd.api.types.is_numeric_dtype(df[request.color_column]):
                value_column = request.color_column
            elif request.y_column and pd.api.types.is_numeric_dtype(df[request.y_column]):
                value_column = request.y_column
            else:
                # Find a numeric column to use as values
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                else:
                    raise ValueError("No numeric column found for treemap values")
            
            # Aggregate data by path column
            treemap_data = df.groupby(path_column)[value_column].sum().reset_index()
            
            # Create treemap
            fig = px.treemap(
                treemap_data,
                path=[path_column],
                values=value_column,
                title=request.title or f"Treemap: {value_column} by {path_column}"
            )
            
            # Update layout for better readability
            fig.update_layout(
                margin=dict(t=50, l=25, r=25, b=25)
            )
            
            return fig
        else:
            # If no specific column specified, try to find suitable columns
            # Look for categorical columns and numeric columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                path_column = categorical_cols[0]
                value_column = numeric_cols[0]
                
                # Aggregate data
                treemap_data = df.groupby(path_column)[value_column].sum().reset_index()
                
                # Create treemap
                fig = px.treemap(
                    treemap_data,
                    path=[path_column],
                    values=value_column,
                    title=request.title or f"Treemap: {value_column} by {path_column}"
                )
                
                # Update layout
                fig.update_layout(
                    margin=dict(t=50, l=25, r=25, b=25)
                )
                
                return fig
            else:
                raise ValueError("Need at least one categorical column and one numeric column for treemap")
    
    def _create_plotly_dual_axis(self, df: pd.DataFrame, request: VisualizationRequest) -> go.Figure:
        """Create Plotly dual-axis chart with bars and line"""
        from plotly.subplots import make_subplots
        import numpy as np
        
        # For dual-axis charts, we need at least 2 numeric columns
        # x_column = grouping dimension (e.g., Month, Category)
        # y_column = first metric (bars, left axis)
        # color_column = second metric (line, right axis)
        
        if not request.x_column:
            # Try to find a suitable grouping column
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) > 0:
                x_col = datetime_cols[0]
                # Extract month/period from datetime for grouping
                df = df.copy()
                df['Period'] = df[x_col].dt.strftime('%Y-%m')
                x_col = 'Period'
            elif len(categorical_cols) > 0:
                x_col = categorical_cols[0]
            else:
                raise ValueError("No suitable grouping column found for dual-axis chart")
        else:
            x_col = request.x_column
        
        # Find numeric columns for the two y-axes
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for dual-axis chart")
        
        # Determine which columns to use for each axis
        y1_col = None  # Primary axis (bars)
        y2_col = None  # Secondary axis (line)
        
        # Smart column selection based on request and common patterns
        if request.y_column and request.y_column in numeric_cols:
            y1_col = request.y_column
        
        if request.color_column and request.color_column in numeric_cols and request.color_column != y1_col:
            y2_col = request.color_column
        
        # If we don't have both columns, use heuristics
        if not y1_col or not y2_col:
            # Look for common patterns in column names
            sales_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['sales', 'revenue', 'amount'])]
            profit_cols = [col for col in numeric_cols if 'profit' in col.lower()]
            
            # Prefer sales for bars (primary) and profit for line (secondary)
            if sales_cols and profit_cols:
                y1_col = sales_cols[0]
                y2_col = profit_cols[0]
            elif len(numeric_cols) >= 2:
                # Fallback to first two numeric columns
                available_cols = [col for col in numeric_cols if col != y1_col]
                if not y1_col:
                    y1_col = numeric_cols[0]
                    available_cols = numeric_cols[1:]
                if not y2_col and available_cols:
                    y2_col = available_cols[0]
        
        # Final validation
        if not y1_col or not y2_col or y1_col == y2_col:
            if len(numeric_cols) >= 2:
                y1_col = numeric_cols[0]
                y2_col = numeric_cols[1]
            else:
                raise ValueError("Need at least 2 different numeric columns for dual-axis chart")
        
        # Prepare data for aggregation
        try:
            # Ensure we have unique columns for aggregation
            agg_columns = [y1_col, y2_col]
            if y1_col == y2_col:
                # If same column, just use it once
                agg_columns = [y1_col]
            
            # Check if x_col is suitable for grouping
            if x_col in df.columns and df[x_col].dtype in ['object', 'category'] or x_col == 'Period':
                agg_data = df.groupby(x_col)[agg_columns].sum().reset_index()
                
                # If we had duplicate columns, duplicate the result
                if y1_col == y2_col and len(agg_columns) == 1:
                    agg_data[f'{y1_col}_line'] = agg_data[y1_col]
                    y2_col = f'{y1_col}_line'
            else:
                # If x_col is not suitable for grouping, use data as-is
                agg_data = df[[x_col, y1_col, y2_col]].copy()
                
        except Exception as e:
            # If grouping fails, use data as-is
            try:
                agg_data = df[[x_col, y1_col, y2_col]].copy()
            except KeyError:
                # If columns don't exist, create minimal data
                agg_data = pd.DataFrame({
                    x_col: df[x_col] if x_col in df.columns else range(len(df)),
                    y1_col: df[y1_col] if y1_col in df.columns else [1] * len(df),
                    y2_col: df[y2_col] if y2_col in df.columns else [1] * len(df)
                })
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart on primary y-axis (left)
        fig.add_trace(
            go.Bar(
                x=agg_data[x_col],
                y=agg_data[y1_col],
                name=y1_col,
                marker_color='lightblue',  # Default color, can be enhanced
                yaxis='y'
            ),
            secondary_y=False
        )
        
        # Add line chart on secondary y-axis (right)
        fig.add_trace(
            go.Scatter(
                x=agg_data[x_col],
                y=agg_data[y2_col],
                mode='lines+markers',
                name=y2_col,
                line=dict(color='red', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        # Update layout and axes
        fig.update_layout(
            title=request.title or f"Dual-Axis Chart: {y1_col} and {y2_col}",
            xaxis_title=x_col,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text=y1_col, secondary_y=False)
        fig.update_yaxes(title_text=y2_col, secondary_y=True)
        
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
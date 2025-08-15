# Visualization Features

## Overview

The CSV Data Analyst now includes comprehensive visualization capabilities that allow users to create charts and graphs through natural language queries. The system automatically detects when a user requests a visualization and generates appropriate charts using either Plotly (default) or Matplotlib.

## Supported Chart Types

### 1. Bar Charts

- **Keywords**: "bar chart", "bar plot", "column chart"
- **Use cases**: Comparing categories, showing counts or sums by group
- **Example**: "Create a bar chart of sales by region"

### 2. Line Charts

- **Keywords**: "line chart", "line plot", "trend", "time series"
- **Use cases**: Showing trends over time, continuous data progression
- **Example**: "Show a line chart of sales over time"

### 3. Scatter Plots

- **Keywords**: "scatter plot", "scatter chart", "correlation plot"
- **Use cases**: Exploring relationships between two numeric variables
- **Example**: "Plot price vs quantity as a scatter plot"

### 4. Histograms

- **Keywords**: "histogram", "distribution", "frequency plot"
- **Use cases**: Showing data distribution, frequency analysis
- **Example**: "Create a histogram of age distribution"

### 5. Box Plots

- **Keywords**: "box plot", "box chart", "boxplot"
- **Use cases**: Showing data distribution, identifying outliers
- **Example**: "Show a box plot of sales data"

### 6. Pie Charts

- **Keywords**: "pie chart", "pie plot"
- **Use cases**: Showing proportions of a whole
- **Example**: "Create a pie chart of market share by region"

### 7. Heatmaps

- **Keywords**: "heatmap", "heat map", "correlation matrix"
- **Use cases**: Showing correlations, matrix data visualization
- **Example**: "Generate a heatmap of correlations"

### 8. Area Charts

- **Keywords**: "area chart", "area plot", "filled line"
- **Use cases**: Showing cumulative data, filled trend lines
- **Example**: "Create an area chart of cumulative sales"

## How It Works

### 1. Natural Language Processing

The system uses intelligent parsing to detect visualization requests in user queries:

- Identifies visualization keywords
- Extracts chart type preferences
- Detects column references
- Generates appropriate titles

### 2. Automatic Chart Selection

When specific chart types aren't mentioned, the system intelligently selects appropriate visualizations based on:

- Data types (numeric vs categorical)
- Number of variables requested
- Common visualization patterns

### 3. Smart Column Detection

The system automatically identifies relevant columns from your query:

- Matches column names mentioned in the query
- Suggests appropriate axes assignments
- Handles both exact and partial column name matches

## Usage Examples

### Basic Visualization Requests

```
"Show me a chart of sales data"
"Plot the distribution of prices"
"Create a graph showing revenue by quarter"
```

### Specific Chart Type Requests

```
"Create a bar chart of customer counts by region"
"Generate a scatter plot of price vs quantity"
"Show a histogram of order amounts"
```

### Column-Specific Requests

```
"Plot sales on the y-axis and date on the x-axis"
"Create a line chart with time as x and revenue as y"
"Show correlation between age and income as a scatter plot"
```

## Technical Implementation

### Components

1. **VisualizationParser**: Analyzes natural language queries for visualization intent
2. **ChartGenerator**: Creates charts using Plotly or Matplotlib
3. **LangChain Tool**: Integrates visualization capabilities with the AI agent
4. **Streamlit Display**: Renders charts in the web interface

### Libraries Used

- **Plotly**: Interactive charts (default)
- **Matplotlib**: Static charts (alternative)
- **Pandas**: Data manipulation and analysis

### Integration

The visualization system is fully integrated with the LangChain pandas DataFrame agent, allowing seamless combination of data analysis and visualization in a single query.

## Configuration

### Default Settings

- **Default Library**: Plotly (configurable via `DEFAULT_CHART_LIBRARY` in config)
- **Chart Size**: Optimized for Streamlit container width
- **Color Schemes**: Uses library defaults with good contrast

### Customization

Users can specify preferences in their queries:

- "Create a matplotlib bar chart..." (forces Matplotlib)
- "Generate an interactive scatter plot..." (ensures Plotly)

## Error Handling

The system gracefully handles various error scenarios:

- **Invalid Data**: Clear error messages for unsuitable data types
- **Missing Columns**: Suggestions for available columns
- **Chart Type Mismatches**: Automatic fallback to appropriate chart types
- **Rendering Issues**: Fallback between Plotly and Matplotlib

## Performance

- **Fast Parsing**: Efficient regex-based query analysis
- **Optimized Rendering**: Charts are generated only when requested
- **Memory Efficient**: Visualizations are stored temporarily and cleaned up appropriately
- **Responsive UI**: Charts are displayed with proper container sizing

## Future Enhancements

Potential future improvements include:

- Custom color schemes and styling
- Advanced chart types (violin plots, treemaps, etc.)
- Export functionality for charts
- Animation support for time-series data
- Dashboard-style multi-chart layouts

"""
CSV Data Analyst - Main Streamlit Application

This application provides a conversational interface for analyzing CSV data
using natural language queries powered by LangChain and Google Gemini.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import io
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import agent components
from src.agents.csv_analyst_agent import CSVAnalystAgent, AgentManager, AgentResponse
from src.config import Config, ConfigurationManager, config_manager
from src.components.conversation_ui import ConversationManager, ChatInterface, ConversationDisplay
from src.utils.error_handler import error_handler, ValidationUtils, GracefulDegradation

# Configure Streamlit page
st.set_page_config(
    page_title="CSV Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables for the application with enhanced conversation management."""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    
    if 'dataframe_info' not in st.session_state:
        st.session_state.dataframe_info = None
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = AgentManager()
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    # Enhanced conversation management
    if 'conversation_metadata' not in st.session_state:
        st.session_state.conversation_metadata = {
            'session_start': datetime.now(),
            'total_queries': 0,
            'successful_queries': 0,
            'visualizations_created': 0
        }
    
    if 'ui_preferences' not in st.session_state:
        st.session_state.ui_preferences = {
            'show_execution_details': False,
            'conversation_display_mode': 'chronological',  # 'chronological' or 'reverse'
            'auto_scroll_to_bottom': True
        }

def initialize_agent() -> tuple[bool, str]:
    """
    Initialize the CSV analyst agent with Google Gemini integration.
    
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Validate API configuration first
        is_api_valid, api_error = ValidationUtils.validate_api_configuration()
        if not is_api_valid:
            error_info = error_handler.handle_error(
                ValueError(api_error), 
                "Agent initialization - API validation"
            )
            return False, error_info.user_message
        
        # Check if API key is configured
        config = Config()
        is_valid, error_msg = st.session_state.agent_manager.validate_configuration()
        
        if not is_valid:
            error_info = error_handler.handle_error(
                ValueError(error_msg), 
                "Agent initialization - configuration validation"
            )
            return False, error_info.user_message
        
        # Initialize the agent
        st.session_state.agent = st.session_state.agent_manager.initialize_agent()
        st.session_state.agent_initialized = True
        
        return True, ""
        
    except Exception as e:
        error_info = error_handler.handle_error(e, "Agent initialization")
        st.session_state.agent_initialized = False
        return False, error_info.user_message

def validate_csv_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded CSV file format and content using enhanced validation.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        return ValidationUtils.validate_csv_file(uploaded_file)
    except Exception as e:
        error_info = error_handler.handle_error(e, "CSV file validation")
        return False, error_info.user_message

def render_file_upload() -> Optional[pd.DataFrame]:
    """
    Render file upload component with validation.
    
    Returns:
        DataFrame if file is successfully uploaded and parsed, None otherwise
    """
    st.subheader("📁 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file to analyze",
        type=['csv'],
        help="Upload a CSV file (max 50MB) to start analyzing your data"
    )
    
    if uploaded_file is not None:
        # Validate the uploaded file
        is_valid, error_message = validate_csv_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_message}")
            return None
        
        # Check if this is a new file or the same file
        if (st.session_state.uploaded_file is None or 
            st.session_state.uploaded_file.name != uploaded_file.name):
            
            try:
                # Parse CSV file
                df = pd.read_csv(uploaded_file)
                
                # Validate the DataFrame
                is_df_valid, df_error = ValidationUtils.validate_dataframe(df)
                if not is_df_valid:
                    error_info = error_handler.handle_error(
                        ValueError(df_error), 
                        "DataFrame validation after CSV parsing"
                    )
                    error_handler.display_error(error_info)
                    return None
                
                # Store in session state
                st.session_state.uploaded_file = uploaded_file
                st.session_state.dataframe = df
                
                # Initialize agent if not already done
                if not st.session_state.agent_initialized:
                    agent_success, agent_error = initialize_agent()
                    if not agent_success:
                        st.error(f"❌ {agent_error}")
                        # Check service availability for graceful degradation
                        services = GracefulDegradation.check_service_availability()
                        if not services['internet']:
                            st.info("🌐 No internet connection detected. You can still preview your data.")
                        elif not services['google_api']:
                            st.info("🤖 AI service unavailable. You can still preview your data.")
                        return df
                
                # Load dataframe into agent
                if st.session_state.agent:
                    try:
                        st.session_state.agent.load_dataframe(df)
                        st.success(f"✅ Successfully loaded {uploaded_file.name} and initialized AI agent")
                    except Exception as e:
                        error_info = error_handler.handle_error(e, "Loading DataFrame into agent")
                        error_handler.display_error(error_info)
                        # Still return the DataFrame for basic functionality
                        return df
                else:
                    st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Clear conversation history for new file
                st.session_state.conversation_history = []
                
                return df
                
            except Exception as e:
                error_info = error_handler.handle_error(e, "CSV file parsing and processing")
                error_handler.display_error(error_info)
                return None
    
    return st.session_state.dataframe

def render_data_preview(df: pd.DataFrame) -> None:
    """
    Render data preview component showing basic information about the dataset.
    
    Args:
        df: Pandas DataFrame to preview
    """
    if df is None:
        return
    
    st.subheader("📋 Data Preview")
    
    # Basic info in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum()
        st.metric("Memory Usage", f"{memory_usage / 1024 / 1024:.1f} MB")
    
    # Show first few rows
    st.write("**First 5 rows:**")
    st.dataframe(df.head(), use_container_width=True)
    
    # Show column information
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)

def update_conversation_metadata(success: bool = True, has_visualization: bool = False) -> None:
    """Update conversation metadata using the ConversationManager."""
    ConversationManager.update_conversation_metadata(success, has_visualization)

def render_conversation_stats() -> None:
    """Render conversation statistics using the ConversationManager."""
    ConversationManager._render_conversation_stats()

def export_conversation_history() -> str:
    """Export conversation history using the ConversationManager."""
    return ConversationManager.export_conversation_history()

def display_visualization(visualization_result) -> None:
    """
    Display a visualization result in Streamlit.
    
    Args:
        visualization_result: VisualizationResult object or visualization data
    """
    try:
        # Initialize chart counter in session state
        if 'chart_counter' not in st.session_state:
            st.session_state.chart_counter = 0
        
        # Increment counter for each chart
        st.session_state.chart_counter += 1
        # Handle VisualizationResult object
        if hasattr(visualization_result, 'chart_data') and hasattr(visualization_result, 'library'):
            chart_data = visualization_result.chart_data
            library = visualization_result.library
            
            if not visualization_result.success:
                st.error(f"Visualization error: {visualization_result.error_message}")
                return
        else:
            # Handle direct chart data (fallback)
            chart_data = visualization_result
            library = "unknown"
        
        if chart_data is None:
            st.warning("No visualization data available")
            return
        
        # Generate unique key for chart to avoid ID conflicts
        import time
        import random
        import uuid
        
        # Use multiple sources for uniqueness including session counter
        timestamp = int(time.time() * 1000000)
        random_num = random.randint(1000, 9999)
        unique_id = str(uuid.uuid4())[:8]
        counter = st.session_state.chart_counter
        chart_key = f"chart_{counter}_{timestamp}_{random_num}_{unique_id}"
        
        # Display based on library type
        if library == "plotly" or hasattr(chart_data, 'show'):
            # Plotly figure with unique key
            st.plotly_chart(chart_data, use_container_width=True, key=chart_key)
        elif library == "matplotlib" or hasattr(chart_data, 'savefig'):
            # Matplotlib figure
            st.pyplot(chart_data)
        else:
            # Try to detect and display
            try:
                # Try plotly first with unique key
                st.plotly_chart(chart_data, use_container_width=True, key=chart_key)
            except:
                try:
                    # Try matplotlib
                    st.pyplot(chart_data)
                except:
                    st.error("Unable to display visualization - unsupported format")
                    
    except Exception as e:
        st.error(f"Error displaying visualization: {str(e)}")


def render_conversation_history() -> None:
    """Render the conversation history using the new ConversationManager component."""
    ConversationManager.render_conversation_history()

def render_chat_interface() -> Optional[str]:
    """
    Render chat interface using the new ChatInterface component.
    
    Returns:
        User query string if submitted, None otherwise
    """
    return ChatInterface.render_chat_input()

def main():
    """Main application function with comprehensive error handling."""
    try:
        # Perform startup validation on first run
        if 'startup_validation_performed' not in st.session_state:
            with st.spinner("🔍 Performing startup validation..."):
                success, validation_report = config_manager.perform_startup_validation()
                st.session_state.startup_validation_performed = True
                st.session_state.startup_validation_success = success
                st.session_state.startup_validation_report = validation_report
                
                if not success:
                    st.error("⚠️ Startup validation found issues. Check the sidebar for details.")
        
        # Initialize session state
        initialize_session_state()
        
        # Check service availability for graceful degradation
        services = GracefulDegradation.check_service_availability()
        if not services['internet'] or not services['google_api']:
            st.warning("⚠️ **Limited Functionality**: Some services are unavailable")
            with st.expander("ℹ️ What's available in limited mode"):
                GracefulDegradation.display_degraded_mode_info()
        
        # App header
        st.title("📊 CSV Data Analyst")
        st.markdown("Analyze your CSV data using natural language queries powered by AI")
        
        # Sidebar for file upload and data info
        with st.sidebar:
            # Configuration status section
            if not st.session_state.get('startup_validation_success', True):
                st.header("⚠️ Configuration Issues")
                with st.expander("View Validation Report", expanded=False):
                    report = st.session_state.get('startup_validation_report', {})
                    summary = report.get('summary', {})
                    
                    st.write(f"**Status**: {summary.get('overall_status', 'Unknown')}")
                    st.write(f"**Checks**: {summary.get('successful_checks', 0)}/{summary.get('total_checks', 0)} passed")
                    
                    failed_checks = [check for check in report.get('checks', []) if check.get('status') == 'FAIL']
                    if failed_checks:
                        st.write("**Issues found**:")
                        for check in failed_checks[:3]:  # Show first 3 issues
                            st.write(f"• {check.get('message', 'Unknown issue')}")
                            if check.get('suggestions'):
                                st.write(f"  → {check['suggestions'][0]}")
                        
                        if len(failed_checks) > 3:
                            st.write(f"... and {len(failed_checks) - 3} more issues")
                    
                    if st.button("🔄 Re-run Validation"):
                        st.session_state.startup_validation_performed = False
                        st.rerun()
            
            st.header("🔧 Data Management")
            
            # File upload section
            dataframe = render_file_upload()
            
            # Agent status section
            st.header("🤖 AI Agent Status")
            
            # Check API key configuration with enhanced validation
            is_api_valid, api_error = ValidationUtils.validate_api_configuration()
            if is_api_valid:
                st.success("✅ Google API Key configured")
            else:
                st.error(f"❌ API Configuration Issue: {api_error}")
                st.info("💡 The app will run in limited mode until the API key is properly configured.")
                
                # Show specific guidance based on the error
                if "not set" in api_error:
                    st.markdown("""
                    **To fix this:**
                    1. Create a `.env` file in your project root
                    2. Add: `GOOGLE_API_KEY=your_actual_api_key_here`
                    3. Restart the application
                    """)
                elif "invalid" in api_error or "format" in api_error:
                    st.markdown("""
                    **To fix this:**
                    1. Check your API key format (should start with 'AIza' or 'ya29')
                    2. Verify you copied the complete key without extra spaces
                    3. Generate a new key if needed from Google Cloud Console
                    """)
            
            # Agent initialization status
            if st.session_state.agent_initialized:
                st.success("✅ AI Agent initialized")
                if st.session_state.agent and st.session_state.agent.is_ready():
                    st.success("✅ Ready for analysis")
                else:
                    st.warning("⚠️ Waiting for data. Please upload a valid CSV file.")
            else:
                st.warning("⚠️ AI Agent not initialized. Please check your API key and try re-uploading your CSV file.")
            
            # Data preview in sidebar if data is loaded
            if dataframe is not None:
                st.success("✅ Data loaded successfully!")
                with st.expander("📊 Quick Stats"):
                    st.write(f"**Shape:** {dataframe.shape[0]} rows × {dataframe.shape[1]} columns")
                    st.write(f"**Columns:** {', '.join(dataframe.columns[:3])}{'...' if len(dataframe.columns) > 3 else ''}")
                    
                    # Show agent-specific data info if available
                    if st.session_state.agent:
                        agent_info = st.session_state.agent.get_dataframe_info()
                        if agent_info:
                            st.write(f"**Memory Usage:** {agent_info['memory_usage']}")
            
            # Conversation management section
            if st.session_state.conversation_history:
                st.markdown("### 💬 Conversation")
                
                # Show conversation stats
                render_conversation_stats()
                
                # Conversation management buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Clear", help="Clear conversation history"):
                        ConversationManager.clear_conversation()
                        st.rerun()
                
                with col2:
                    # Export conversation button
                    if st.button("📥 Export", help="Export conversation history"):
                        export_text = export_conversation_history()
                        st.download_button(
                            label="💾 Download",
                            data=export_text,
                            file_name=f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download conversation as text file"
                        )
            
            # Configuration summary section
            with st.expander("⚙️ Configuration", expanded=False):
                config = Config()
                config_summary = config.get_configuration_summary()
                
                st.write("**API Configuration**:")
                st.write(f"• API Key: {'✅ Configured' if config_summary['api_key_configured'] else '❌ Missing'}")
                if config_summary['api_key_configured']:
                    st.write(f"• Format: {'✅ Valid' if config_summary['api_key_format_valid'] else '⚠️ Invalid'}")
                
                st.write("**Limits**:")
                st.write(f"• Max file size: {config_summary['max_file_size_mb']:.0f} MB")
                st.write(f"• Conversation history: {config_summary['max_conversation_history']}")
                
                st.write("**Features**:")
                st.write(f"• Chart library: {config_summary['chart_library']}")
                st.write(f"• Debug mode: {'✅ On' if config_summary['debug_mode'] else '❌ Off'}")
                st.write(f"• Code execution: {'✅ Enabled' if config_summary['code_execution_enabled'] else '❌ Disabled'}")
                
                if st.button("🔍 Run Health Check"):
                    health_status = config.get_health_status()
                    st.json(health_status)
        
        # Main content area
        if st.session_state.dataframe is not None:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["💬 Chat", "📋 Data Preview"])
            
            with tab1:
                # Disable chat input if agent is not ready or API is missing
                agent_ready = st.session_state.agent and st.session_state.agent.is_ready()
                api_key_ok = bool(Config().GOOGLE_API_KEY)
                if not agent_ready or not api_key_ok:
                    st.warning("⚠️ The chat is disabled because the AI agent is not ready or the API key is missing. Please check the sidebar for details.")
                else:
                    # Chat interface
                    user_query = render_chat_interface()
                    
                    # Debug information (can be removed after testing)
                    with st.expander("🔧 Debug Info", expanded=False):
                        st.write(f"is_processing: {st.session_state.is_processing}")
                        st.write(f"agent_initialized: {st.session_state.agent_initialized}")
                        st.write(f"dataframe loaded: {st.session_state.dataframe is not None}")
                        st.write(f"agent exists: {st.session_state.agent is not None}")
                        if st.session_state.agent:
                            st.write(f"agent.dataframe exists: {st.session_state.agent.dataframe is not None}")
                            st.write(f"agent.agent exists: {st.session_state.agent.agent is not None}")
                            st.write(f"agent.is_ready(): {st.session_state.agent.is_ready()}")
                        
                        # Check API key
                        config = Config()
                        st.write(f"API key configured: {bool(config.GOOGLE_API_KEY)}")
                        if config.GOOGLE_API_KEY:
                            st.write(f"API key starts with: {config.GOOGLE_API_KEY[:10]}...")
                        
                        if st.button("Reset Processing State"):
                            st.session_state.is_processing = False
                            st.rerun()
                        
                        if st.button("Reinitialize Agent"):
                            if st.session_state.dataframe is not None:
                                try:
                                    agent_success, agent_error = initialize_agent()
                                    if agent_success and st.session_state.agent:
                                        st.session_state.agent.load_dataframe(st.session_state.dataframe)
                                        st.success("Agent reinitialized successfully!")
                                    else:
                                        st.error(f"Failed to reinitialize: {agent_error}")
                                except Exception as e:
                                    st.error(f"Error reinitializing: {str(e)}")
                                st.rerun()
                            else:
                                st.warning("No dataframe loaded")
                    
                    # Process query with agent
                    if user_query:
                        # Validate user query first
                        is_query_valid, query_error = ValidationUtils.validate_user_query(user_query)
                        if not is_query_valid:
                            error_info = error_handler.handle_error(
                                ValueError(query_error), 
                                "User query validation"
                            )
                            error_handler.display_error(error_info)
                            st.session_state.is_processing = False
                            return
                        
                        with st.spinner("🤔 Analyzing your question..."):
                            # Check service availability before processing
                            services = GracefulDegradation.check_service_availability()
                            
                            if not services['internet']:
                                error_info = error_handler.handle_error(
                                    ConnectionError("No internet connection"), 
                                    "Service availability check"
                                )
                                error_handler.display_error(error_info)
                                GracefulDegradation.display_degraded_mode_info()
                                st.session_state.is_processing = False
                                return
                            
                            if st.session_state.agent and st.session_state.agent.is_ready():
                                try:
                                    # Process query with the agent
                                    agent_response = st.session_state.agent.process_query(user_query)
                                    
                                    if agent_response.success:
                                        response = agent_response.text_response
                                        has_viz = agent_response.visualization is not None
                                        
                                        # Add assistant response to conversation with enhanced metadata
                                        ConversationManager.add_message(
                                            'assistant',
                                            response,
                                            visualization=agent_response.visualization,
                                            execution_details=agent_response.execution_details,
                                            success=True,
                                            has_visualization=has_viz,
                                            response_length=len(response)
                                        )
                                        
                                        # Update conversation metadata
                                        update_conversation_metadata(success=True, has_visualization=has_viz)
                                        
                                    else:
                                        # Handle agent error response with enhanced error handling
                                        error_info = error_handler.handle_error(
                                            RuntimeError(agent_response.error_message), 
                                            "Agent query processing"
                                        )
                                        
                                        ConversationManager.add_message(
                                            'assistant',
                                            error_info.user_message,
                                            error=True,
                                            success=False,
                                            execution_details=agent_response.execution_details if agent_response.execution_details else {},
                                            error_suggestions=error_info.suggestions
                                        )
                                        
                                        # Update conversation metadata
                                        update_conversation_metadata(success=False, has_visualization=False)
                                        
                                except Exception as e:
                                    # Handle unexpected errors with comprehensive error handling
                                    error_info = error_handler.handle_error(e, "Agent query processing - unexpected error")
                                    
                                    ConversationManager.add_message(
                                        'assistant',
                                        error_info.user_message,
                                        error=True,
                                        success=False,
                                        error_suggestions=error_info.suggestions,
                                        error_category=error_info.category.value
                                    )
                                    update_conversation_metadata(success=False, has_visualization=False)
                                    
                            else:
                                # Agent not ready - provide specific guidance
                                if not st.session_state.agent_initialized:
                                    error_msg = "🤖 AI agent is not initialized. Please check your API key configuration."
                                elif not st.session_state.agent:
                                    error_msg = "🤖 AI agent is not available. Please try reloading your data."
                                elif not st.session_state.agent.is_ready():
                                    error_msg = "📊 AI agent is not ready. Please ensure your data is properly loaded."
                                else:
                                    error_msg = "⚙️ AI agent is in an unknown state. Please try refreshing the page."
                                
                                ConversationManager.add_message(
                                    'assistant',
                                    error_msg,
                                    error=True,
                                    success=False,
                                    error_suggestions=[
                                        "Check your Google API key configuration",
                                        "Try re-uploading your CSV file",
                                        "Refresh the page and try again",
                                        "Contact support if the problem persists"
                                    ]
                                )
                                
                                # Update conversation metadata
                                update_conversation_metadata(success=False, has_visualization=False)
                            
                            # Reset processing state
                            st.session_state.is_processing = False
                            st.session_state.current_query = ""
                            
                            # Rerun to update the display
                            st.rerun()
                    
                    # Conversation display preferences
                    ChatInterface.render_display_preferences()
                    
                    # Display conversation history
                    render_conversation_history()
            
            with tab2:
                # Data preview
                render_data_preview(st.session_state.dataframe)
        
        else:
            # Welcome message when no data is loaded
            st.info("👋 Welcome! Please upload a CSV file using the sidebar to get started.")
            
            # Show example of what the app can do
            st.subheader("🚀 What can you do with CSV Data Analyst?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Data Analysis:**
                - Get summary statistics
                - Find patterns and trends
                - Calculate correlations
                - Identify outliers
                """)
            
            with col2:
                st.markdown("""
                **📈 Visualizations:**
                - Generate charts and graphs
                - Create histograms and scatter plots
                - Build comparison charts
                - Visualize distributions
                """)
    except Exception as e:
        # Handle critical application errors
        error_info = error_handler.handle_error(e, "Main application execution")
        st.error("🚨 **Critical Application Error**")
        error_handler.display_error(error_info)
        
        # Provide recovery options
        st.markdown("### 🔧 Recovery Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("📊 View Error History"):
                error_history = error_handler.get_error_history()
                if error_history:
                    st.write("Recent errors:")
                    for i, err in enumerate(error_history[-5:], 1):
                        st.write(f"{i}. {err.category.value}: {err.message}")
                else:
                    st.write("No error history available")

if __name__ == "__main__":
    main()
"""
Conversation UI Components for CSV Data Analyst

This module contains specialized UI components for managing and displaying
conversation history, chat interface, and message formatting.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import io


class ConversationMessage:
    """Data class for conversation messages with enhanced metadata."""
    
    def __init__(self, role: str, content: str, timestamp: datetime = None, **kwargs):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for session state storage."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create ConversationMessage from dictionary."""
        role = data.pop('role')
        content = data.pop('content')
        timestamp = data.pop('timestamp', datetime.now())
        return cls(role, content, timestamp, **data)


class ConversationDisplay:
    """Component for displaying conversation history with enhanced formatting."""
    
    @staticmethod
    def render_message_pair(user_msg: Dict[str, Any], assistant_msg: Dict[str, Any], 
                           show_details: bool = False) -> None:
        """
        Render a user-assistant message pair with enhanced styling.
        
        Args:
            user_msg: User message dictionary
            assistant_msg: Assistant message dictionary
            show_details: Whether to show execution details
        """
        # Create container for the message pair
        with st.container():
            # User message
            ConversationDisplay._render_user_message(user_msg)
            
            # Assistant message
            ConversationDisplay._render_assistant_message(assistant_msg, show_details)
            
            # Add separator
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;'>", 
                       unsafe_allow_html=True)
    
    @staticmethod
    def _render_user_message(message: Dict[str, Any]) -> None:
        """Render user message with styling and smart truncation."""
        timestamp = ConversationDisplay._format_timestamp(message.get('timestamp'))
        content = message['content']
        
        # Handle long messages with expandable content
        max_length = 500
        is_long = len(content) > max_length
        
        if is_long:
            # Create expandable content
            short_content = content[:max_length] + "..."
            with st.container():
                # Enhanced user message styling with truncation
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 12px; border-radius: 10px; margin: 8px 0; 
                           border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="color: #1f77b4;">🙋 You</strong>
                        <small style="color: #666; font-size: 0.8em;">{timestamp}</small>
                    </div>
                    <div style="line-height: 1.5; color: #262730;">{short_content}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add expandable section for full content
                with st.expander("📖 Show full message", expanded=False):
                    st.markdown(f"<div style='line-height: 1.5;'>{content}</div>", unsafe_allow_html=True)
        else:
            # Enhanced user message styling
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 12px; border-radius: 10px; margin: 8px 0; 
                       border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: #1f77b4;">🙋 You</strong>
                    <small style="color: #666; font-size: 0.8em;">{timestamp}</small>
                </div>
                <div style="line-height: 1.5; color: #262730;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_assistant_message(message: Dict[str, Any], show_details: bool = False) -> None:
        """Render assistant message with styling, smart truncation, and optional details."""
        timestamp = ConversationDisplay._format_timestamp(message.get('timestamp'))
        
        # Determine styling based on success/error
        if message.get('error'):
            border_color = "#ff4444"
            bg_color = "#ffe6e6"
            icon = "❌"
            title_color = "#cc0000"
            content = message['content']
            
            # Add error suggestions if available
            if message.get('error_suggestions'):
                content += "\n\n**💡 Suggestions:**"
                for suggestion in message['error_suggestions']:
                    content += f"\n• {suggestion}"
        else:
            border_color = "#28a745"
            bg_color = "#e8f5e8"
            icon = "🤖"
            title_color = "#28a745"
            content = message['content']
        
        # Handle long messages with expandable content
        max_length = 800
        is_long = len(content) > max_length
        
        if is_long:
            # Create expandable content
            short_content = content[:max_length] + "..."
            with st.container():
                # Enhanced assistant message styling with truncation
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 12px; border-radius: 10px; margin: 8px 0; 
                           border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="color: {title_color};">{icon} Assistant</strong>
                        <small style="color: #666; font-size: 0.8em;">{timestamp}</small>
                    </div>
                    <div style="line-height: 1.5; color: #262730;">{short_content}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add expandable section for full content
                with st.expander("📖 Show full response", expanded=False):
                    st.markdown(f"<div style='line-height: 1.5;'>{content}</div>", unsafe_allow_html=True)
        else:
            # Enhanced assistant message styling
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 12px; border-radius: 10px; margin: 8px 0; 
                       border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: {title_color};">{icon} Assistant</strong>
                    <small style="color: #666; font-size: 0.8em;">{timestamp}</small>
                </div>
                <div style="line-height: 1.5; color: #262730;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show execution details if requested
        if show_details and message.get('execution_details'):
            ConversationDisplay._render_execution_details(message)
        
        # Display visualization if present
        if message.get('visualization'):
            st.markdown("### 📊 Visualization")
            ConversationDisplay._display_visualization(message['visualization'])
    
    @staticmethod
    def _render_execution_details(message: Dict[str, Any]) -> None:
        """Render execution details in an expandable section."""
        with st.expander("🔍 Execution Details", expanded=False):
            details = message['execution_details']
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                if 'execution_time' in details:
                    st.metric("⏱️ Execution Time", f"{details['execution_time']:.2f}s")
                
                if message.get('has_visualization'):
                    viz_status = "✅ Yes" if message['has_visualization'] else "❌ No"
                    st.metric("📊 Visualization", viz_status)
                
                if message.get('response_length'):
                    st.metric("📝 Response Length", f"{message['response_length']} chars")
            
            with col2:
                if 'timestamp' in details:
                    full_timestamp = details['timestamp']
                    if isinstance(full_timestamp, datetime):
                        full_timestamp = full_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    st.write(f"**📅 Full Timestamp:** {full_timestamp}")
                
                if message.get('message_id') is not None:
                    st.write(f"**🔢 Message ID:** {message['message_id']}")
                
                if message.get('success') is not None:
                    status = "✅ Success" if message['success'] else "❌ Failed"
                    st.write(f"**📊 Status:** {status}")
            
            # Show enhanced query if available
            if 'enhanced_question' in details:
                with st.expander("📝 Enhanced Query"):
                    st.code(details['enhanced_question'], language="text")
    
    @staticmethod
    def _display_visualization(visualization_result) -> None:
        """Display a visualization result in Streamlit."""
        try:
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
            
            # Display based on library type
            if library == "plotly" or hasattr(chart_data, 'show'):
                # Plotly figure
                st.plotly_chart(chart_data, use_container_width=True)
            elif library == "matplotlib" or hasattr(chart_data, 'savefig'):
                # Matplotlib figure
                st.pyplot(chart_data)
            else:
                # Try to detect and display
                try:
                    # Try plotly first
                    st.plotly_chart(chart_data, use_container_width=True)
                except:
                    try:
                        # Try matplotlib
                        st.pyplot(chart_data)
                    except:
                        st.error("Unable to display visualization - unsupported format")
                        
        except Exception as e:
            st.error(f"Error displaying visualization: {str(e)}")
    
    @staticmethod
    def _format_timestamp(timestamp) -> str:
        """Format timestamp for display."""
        if timestamp is None:
            return datetime.now().strftime("%H:%M:%S")
        
        if isinstance(timestamp, str):
            return timestamp
        
        return timestamp.strftime("%H:%M:%S")


class ConversationManager:
    """Component for managing conversation history and preferences."""
    
    @staticmethod
    def render_conversation_history() -> None:
        """Render the complete conversation history with enhanced formatting and filtering."""
        if not st.session_state.conversation_history:
            ConversationManager._render_empty_state()
            return
        
        st.subheader("💬 Conversation History")
        
        # Get display preferences
        show_details = st.session_state.ui_preferences.get('show_execution_details', False)
        display_mode = st.session_state.ui_preferences.get('conversation_display_mode', 'chronological')
        message_filter = st.session_state.ui_preferences.get('message_filter', 'all')
        search_query = st.session_state.ui_preferences.get('search_query', '')
        
        # Apply filtering
        messages = ConversationManager._filter_messages(st.session_state.conversation_history, message_filter, search_query)
        
        if not messages:
            if search_query:
                st.info(f"🔍 No messages found matching '{search_query}'")
            elif message_filter != 'all':
                st.info(f"📋 No messages match the current filter: {message_filter}")
            return
        
        # Show filter/search results info
        if search_query or message_filter != 'all':
            total_messages = len(st.session_state.conversation_history)
            filtered_messages = len(messages)
            st.info(f"📊 Showing {filtered_messages} of {total_messages} messages")
        
        # Apply display order
        if display_mode == 'reverse':
            messages = list(reversed(messages))
        
        # Group messages into pairs (user + assistant)
        message_pairs = ConversationManager._group_messages_into_pairs(messages)
        
        # Display each pair
        for pair in message_pairs:
            if len(pair) == 2:  # Complete pair
                ConversationDisplay.render_message_pair(pair[0], pair[1], show_details)
            else:  # Single message (usually user message waiting for response)
                if pair[0]['role'] == 'user':
                    ConversationDisplay._render_user_message(pair[0])
                else:
                    ConversationDisplay._render_assistant_message(pair[0], show_details)
    
    @staticmethod
    def _render_empty_state() -> None:
        """Render empty state when no conversation history exists."""
        st.info("💬 Start a conversation by asking a question about your data!")
        
        # Show helpful tips
        with st.expander("💡 Tips for better conversations", expanded=True):
            st.markdown("""
            **Great questions to ask:**
            - "What are the basic statistics of this dataset?"
            - "Show me the distribution of [column_name]"
            - "Create a bar chart of the top 10 values"
            - "What correlations exist between numeric columns?"
            - "Are there any missing values or outliers?"
            
            **For visualizations:**
            - Be specific about chart types (bar, line, scatter, histogram)
            - Mention which columns you want to analyze
            - Ask for comparisons between different groups
            """)
    
    @staticmethod
    def _filter_messages(messages: List[Dict[str, Any]], filter_type: str, search_query: str = '') -> List[Dict[str, Any]]:
        """Filter messages based on type and search query."""
        filtered_messages = messages.copy()
        
        # Apply type filter
        if filter_type == 'user_only':
            filtered_messages = [m for m in filtered_messages if m['role'] == 'user']
        elif filter_type == 'assistant_only':
            filtered_messages = [m for m in filtered_messages if m['role'] == 'assistant']
        elif filter_type == 'with_visualizations':
            filtered_messages = [m for m in filtered_messages if m.get('has_visualization') or m.get('visualization')]
        
        # Apply search filter
        if search_query:
            filtered_messages = [
                m for m in filtered_messages 
                if search_query in m['content'].lower()
            ]
        
        return filtered_messages
    
    @staticmethod
    def _group_messages_into_pairs(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group messages into user-assistant pairs for better display."""
        pairs = []
        current_pair = []
        
        for message in messages:
            current_pair.append(message)
            
            # If we have an assistant message, complete the pair
            if message['role'] == 'assistant':
                pairs.append(current_pair)
                current_pair = []
            # If we have two user messages in a row, start a new pair
            elif len(current_pair) > 1 and message['role'] == 'user':
                pairs.append(current_pair[:-1])  # Add previous messages as incomplete pair
                current_pair = [message]  # Start new pair with current message
        
        # Add any remaining messages as incomplete pair
        if current_pair:
            pairs.append(current_pair)
        
        return pairs
    
    @staticmethod
    def render_conversation_controls() -> None:
        """Render conversation management controls."""
        if not st.session_state.conversation_history:
            return
        
        st.markdown("### 💬 Conversation Controls")
        
        # Statistics
        ConversationManager._render_conversation_stats()
        
        # Management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear History", help="Clear all conversation history"):
                ConversationManager.clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("📥 Export", help="Export conversation history"):
                ConversationManager._show_export_options()
    
    @staticmethod
    def _render_conversation_stats() -> None:
        """Render conversation statistics."""
        metadata = st.session_state.conversation_metadata
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", metadata.get('total_queries', 0))
            st.metric("Visualizations", metadata.get('visualizations_created', 0))
        
        with col2:
            total = metadata.get('total_queries', 0)
            successful = metadata.get('successful_queries', 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
            
            session_start = metadata.get('session_start', datetime.now())
            session_duration = datetime.now() - session_start
            st.metric("Session Time", f"{session_duration.seconds // 60}m")
    
    @staticmethod
    def _show_export_options() -> None:
        """Show export options for conversation history."""
        export_text = ConversationManager.export_conversation_history()
        
        st.download_button(
            label="💾 Download as Text",
            data=export_text,
            file_name=f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download conversation as text file"
        )
    
    @staticmethod
    def export_conversation_history() -> str:
        """Export conversation history as formatted text with enhanced statistics."""
        if not st.session_state.conversation_history:
            return "No conversation history to export."
        
        conversation_history = st.session_state.conversation_history
        
        # Header
        export_text = f"CSV Data Analyst - Conversation Export\n"
        export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += f"Total Messages: {len(conversation_history)}\n"
        export_text += "=" * 50 + "\n\n"
        
        # Conversation messages
        for i, message in enumerate(conversation_history, 1):
            timestamp = message.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            role = "USER" if message['role'] == 'user' else "ASSISTANT"
            export_text += f"[{i}] {role} ({timestamp_str}):\n"
            export_text += f"{message['content']}\n"
            
            if message.get('visualization'):
                export_text += "[Visualization was displayed]\n"
            
            if message.get('execution_details'):
                details = message['execution_details']
                if 'execution_time' in details:
                    export_text += f"Execution Time: {details['execution_time']:.2f}s\n"
            
            export_text += "\n" + "-" * 30 + "\n\n"
        
        # Enhanced statistics section
        total_messages = len(conversation_history)
        user_messages = len([m for m in conversation_history if m['role'] == 'user'])
        assistant_messages = len([m for m in conversation_history if m['role'] == 'assistant'])
        visualizations = len([m for m in conversation_history if m.get('has_visualization')])
        successful_queries = len([m for m in conversation_history if m.get('success') is True])
        
        # Calculate average execution time
        execution_times = [
            m.get('execution_details', {}).get('execution_time', 0) 
            for m in conversation_history 
            if m.get('execution_details', {}).get('execution_time')
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        export_text += "\n" + "=" * 50 + "\n"
        export_text += "CONVERSATION STATISTICS\n"
        export_text += "=" * 50 + "\n"
        export_text += f"Total Messages: {total_messages}\n"
        export_text += f"User Messages: {user_messages}\n"
        export_text += f"Assistant Messages: {assistant_messages}\n"
        export_text += f"Visualizations Created: {visualizations}\n"
        export_text += f"Successful Queries: {successful_queries}\n"
        if avg_execution_time > 0:
            export_text += f"Average Execution Time: {avg_execution_time:.2f}s\n"
        
        # Session information
        metadata = st.session_state.conversation_metadata
        if metadata:
            session_start = metadata.get('session_start', datetime.now())
            session_duration = datetime.now() - session_start
            export_text += f"Session Duration: {session_duration.seconds // 60}m {session_duration.seconds % 60}s\n"
            
            success_rate = (metadata.get('successful_queries', 0) / metadata.get('total_queries', 1) * 100) if metadata.get('total_queries', 0) > 0 else 0
            export_text += f"Success Rate: {success_rate:.1f}%\n"
        
        return export_text
    
    @staticmethod
    def clear_conversation() -> None:
        """Clear conversation history and reset metadata."""
        st.session_state.conversation_history = []
        
        # Reset conversation metadata
        st.session_state.conversation_metadata = {
            'session_start': datetime.now(),
            'total_queries': 0,
            'successful_queries': 0,
            'visualizations_created': 0
        }
        
        # Clear agent memory if available
        if st.session_state.get('agent'):
            st.session_state.agent.clear_memory()
    
    @staticmethod
    def add_message(role: str, content: str, **kwargs) -> None:
        """Add a message to conversation history with metadata."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'message_id': len(st.session_state.conversation_history),
            **kwargs
        }
        
        st.session_state.conversation_history.append(message)
    
    @staticmethod
    def update_conversation_metadata(success: bool = True, has_visualization: bool = False) -> None:
        """Update conversation metadata for tracking session statistics."""
        st.session_state.conversation_metadata['total_queries'] += 1
        
        if success:
            st.session_state.conversation_metadata['successful_queries'] += 1
        
        if has_visualization:
            st.session_state.conversation_metadata['visualizations_created'] += 1


class ChatInterface:
    """Component for handling chat input and user interactions."""
    
    @staticmethod
    def render_chat_input() -> Optional[str]:
        """
        Render enhanced chat interface for user input.
        
        Returns:
            User query string if submitted, None otherwise
        """
        st.subheader("💭 Ask a Question")
        
        # Check if data is loaded
        if st.session_state.dataframe is None:
            st.warning("⚠️ Please upload a CSV file first to start asking questions.")
            return None
        
        # Show helpful suggestions if no conversation history
        if not st.session_state.conversation_history:
            ChatInterface._render_suggestions()
        
        # Chat input form
        return ChatInterface._render_input_form()
    
    @staticmethod
    def _render_suggestions() -> None:
        """Render helpful suggestions for first-time users."""
        st.info("💡 **Try asking questions like:**")
        
        suggestions = [
            "What are the basic statistics of this dataset?",
            "Show me the distribution of [column_name]",
            "Create a bar chart of the top 10 values in [column_name]",
            "What are the correlations between numeric columns?",
            "Are there any missing values in the data?"
        ]
        
        # Replace [column_name] with actual column names if available
        if st.session_state.dataframe is not None and len(st.session_state.dataframe.columns) > 0:
            sample_column = st.session_state.dataframe.columns[0]
            suggestions = [s.replace("[column_name]", sample_column) for s in suggestions]
        
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
        st.markdown("---")
    
    @staticmethod
    def _render_input_form() -> Optional[str]:
        """Render the chat input form."""
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_area(
                "What would you like to know about your data?",
                placeholder="e.g., What are the top 5 values in the sales column? Can you create a visualization?",
                disabled=st.session_state.is_processing,
                height=100,
                help="Ask questions about your data or request visualizations. The AI can create charts, analyze patterns, and provide insights."
            )
            
            # Show processing status
            if st.session_state.is_processing:
                st.info("⏳ Processing in progress... Please wait before submitting another query.")
                if st.session_state.current_query:
                    st.markdown(f"**Currently processing:** {st.session_state.current_query}")
            
            # Show helpful hint about button usage
            if not user_query.strip():
                st.info("💡 Type a question above and click Analyze to get insights about your data.")
            
            # Submit button - remove disabled logic to fix the button issue
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                submit_clicked = st.form_submit_button(
                    "🚀 Analyze",
                    type="primary",
                    use_container_width=True
                )
        
        # Handle form submission
        if submit_clicked:
            # Validate input
            if not user_query.strip():
                st.warning("⚠️ Please enter a question before clicking Analyze.")
                return None
            
            # Check if already processing
            if st.session_state.is_processing:
                st.warning("⚠️ Already processing a query. Please wait...")
                return None
            
            # Add user message to conversation
            ConversationManager.add_message(
                'user', 
                user_query,
                word_count=len(user_query.split())
            )
            
            # Set processing state
            st.session_state.is_processing = True
            st.session_state.current_query = user_query
            
            return user_query
        
        return None
    
    @staticmethod
    def render_display_preferences() -> None:
        """Render conversation display preferences with enhanced options."""
        if not st.session_state.conversation_history:
            return
        
        with st.expander("⚙️ Display Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.ui_preferences['show_execution_details'] = st.checkbox(
                    "Show execution details",
                    value=st.session_state.ui_preferences.get('show_execution_details', False),
                    help="Show detailed execution information for each response"
                )
                
                # Message filtering options
                filter_option = st.selectbox(
                    "Show messages",
                    options=['all', 'user_only', 'assistant_only', 'with_visualizations'],
                    index=0,
                    help="Filter which messages to display"
                )
                st.session_state.ui_preferences['message_filter'] = filter_option
            
            with col2:
                display_mode = st.selectbox(
                    "Display order",
                    options=['chronological', 'reverse'],
                    index=0 if st.session_state.ui_preferences.get('conversation_display_mode', 'chronological') == 'chronological' else 1,
                    help="Choose how to display conversation messages"
                )
                st.session_state.ui_preferences['conversation_display_mode'] = display_mode
                
                # Search functionality
                search_query = st.text_input(
                    "Search messages",
                    placeholder="Search in conversation...",
                    help="Search for specific content in the conversation"
                )
                if search_query:
                    st.session_state.ui_preferences['search_query'] = search_query.lower()
                else:
                    st.session_state.ui_preferences.pop('search_query', None)
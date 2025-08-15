"""
Comprehensive Error Handling System for CSV Data Analyst

This module provides centralized error handling, user-friendly error messages,
validation utilities, and graceful degradation capabilities.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import traceback
import logging
from datetime import datetime
import requests
import os


class ErrorCategory(Enum):
    """Categories of errors for better handling and user messaging"""
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    FILE_ERROR = "file_error"
    DATA_ERROR = "data_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Structured error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    suggestions: List[str]
    technical_details: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ErrorHandler:
    """
    Centralized error handling system with user-friendly messaging
    and graceful degradation capabilities.
    """
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        Handle an error and return structured error information.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            ErrorInfo object with structured error details
        """
        error_info = self._classify_error(error, context)
        
        # Log the error
        self.logger.error(f"{error_info.category.value}: {error_info.message}")
        if error_info.technical_details:
            self.logger.debug(f"Technical details: {error_info.technical_details}")
        
        # Store in history
        self.error_history.append(error_info)
        
        return error_info
    
    def _classify_error(self, error: Exception, context: str) -> ErrorInfo:
        """Classify error and generate appropriate messaging"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # API-related errors
        if any(keyword in error_str for keyword in ['api key', 'authentication', 'unauthorized', '401', '403']):
            return self._create_api_auth_error(error, context)
        elif any(keyword in error_str for keyword in ['quota', 'rate limit', '429', 'too many requests']):
            return self._create_api_quota_error(error, context)
        elif any(keyword in error_str for keyword in ['google', 'gemini', 'genai']):
            return self._create_api_general_error(error, context)
        
        # Network-related errors
        elif any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'dns', 'unreachable']):
            return self._create_network_error(error, context)
        elif 'requests' in error_str or isinstance(error, requests.RequestException):
            return self._create_network_error(error, context)
        
        # File-related errors
        elif any(keyword in error_str for keyword in ['file', 'csv', 'upload', 'parse', 'read']):
            return self._create_file_error(error, context)
        elif isinstance(error, (FileNotFoundError, PermissionError, pd.errors.EmptyDataError)):
            return self._create_file_error(error, context)
        
        # Data-related errors
        elif any(keyword in error_str for keyword in ['dataframe', 'column', 'index', 'dtype']):
            return self._create_data_error(error, context)
        elif isinstance(error, (KeyError, ValueError, IndexError)) and 'dataframe' in context.lower():
            return self._create_data_error(error, context)
        
        # Validation errors
        elif any(keyword in error_str for keyword in ['validation', 'invalid', 'required', 'missing']):
            return self._create_validation_error(error, context)
        
        # Processing errors
        elif any(keyword in error_str for keyword in ['agent', 'langchain', 'processing', 'execution']):
            return self._create_processing_error(error, context)
        
        # Configuration errors
        elif any(keyword in error_str for keyword in ['config', 'environment', 'setting']):
            return self._create_configuration_error(error, context)
        
        # Default to system error
        else:
            return self._create_system_error(error, context)
    
    def _create_api_auth_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for API authentication issues"""
        return ErrorInfo(
            category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.HIGH,
            message=f"API authentication failed: {str(error)}",
            user_message="🔑 API Key Issue: There's a problem with your Google API key.",
            suggestions=[
                "Check that your GOOGLE_API_KEY environment variable is set correctly",
                "Verify your API key is valid and hasn't expired",
                "Ensure your API key has access to the Gemini API",
                "Try regenerating your API key in the Google Cloud Console"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="API_AUTH_001"
        )
    
    def _create_api_quota_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for API quota/rate limit issues"""
        return ErrorInfo(
            category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"API quota/rate limit exceeded: {str(error)}",
            user_message="⏱️ API Limit Reached: You've hit the API usage limit.",
            suggestions=[
                "Wait a few minutes before trying again",
                "Check your API quota in the Google Cloud Console",
                "Consider upgrading your API plan if you need higher limits",
                "Try breaking down complex queries into smaller parts"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="API_QUOTA_001"
        )
    
    def _create_api_general_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for general API issues"""
        return ErrorInfo(
            category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"API service error: {str(error)}",
            user_message="🤖 AI Service Issue: The AI service is experiencing problems.",
            suggestions=[
                "Try your request again in a few moments",
                "Check if the Google Gemini service is operational",
                "Simplify your query if it's very complex",
                "Contact support if the problem persists"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="API_GENERAL_001"
        )
    
    def _create_network_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for network connectivity issues"""
        return ErrorInfo(
            category=ErrorCategory.NETWORK_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"Network connectivity issue: {str(error)}",
            user_message="🌐 Connection Problem: Unable to connect to the AI service.",
            suggestions=[
                "Check your internet connection",
                "Try refreshing the page",
                "Verify that your firewall isn't blocking the connection",
                "If using a VPN, try disconnecting temporarily"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="NETWORK_001"
        )
    
    def _create_file_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for file-related issues"""
        return ErrorInfo(
            category=ErrorCategory.FILE_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"File processing error: {str(error)}",
            user_message="📁 File Problem: There's an issue with your CSV file.",
            suggestions=[
                "Ensure your file is a valid CSV format",
                "Check that the file isn't corrupted or empty",
                "Verify the file size is under 50MB",
                "Try opening the file in a spreadsheet program to check for issues"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="FILE_001"
        )
    
    def _create_data_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for data processing issues"""
        return ErrorInfo(
            category=ErrorCategory.DATA_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"Data processing error: {str(error)}",
            user_message="📊 Data Issue: There's a problem processing your data.",
            suggestions=[
                "Check that your data has the expected columns",
                "Verify column names don't have special characters",
                "Ensure numeric columns contain valid numbers",
                "Try asking about a different column or aspect of the data"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="DATA_001"
        )
    
    def _create_validation_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for validation issues"""
        return ErrorInfo(
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            message=f"Validation error: {str(error)}",
            user_message="✅ Input Issue: Please check your input.",
            suggestions=[
                "Make sure all required fields are filled",
                "Check that your question is clear and specific",
                "Verify you've uploaded a CSV file before asking questions",
                "Try rephrasing your question"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="VALIDATION_001"
        )
    
    def _create_processing_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for processing/agent issues"""
        return ErrorInfo(
            category=ErrorCategory.PROCESSING_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message=f"Processing error: {str(error)}",
            user_message="⚙️ Processing Issue: The AI couldn't process your request.",
            suggestions=[
                "Try rephrasing your question more simply",
                "Break complex requests into smaller parts",
                "Check that your question relates to the uploaded data",
                "Try asking about basic statistics first"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="PROCESSING_001"
        )
    
    def _create_configuration_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for configuration issues"""
        return ErrorInfo(
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            message=f"Configuration error: {str(error)}",
            user_message="⚙️ Setup Issue: The application isn't configured properly.",
            suggestions=[
                "Check your environment variables are set correctly",
                "Verify the .env file exists and has the right values",
                "Restart the application after making configuration changes",
                "Contact your administrator if you can't access configuration"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}",
            error_code="CONFIG_001"
        )
    
    def _create_system_error(self, error: Exception, context: str) -> ErrorInfo:
        """Create error info for general system issues"""
        return ErrorInfo(
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.HIGH,
            message=f"System error: {str(error)}",
            user_message="🔧 System Issue: An unexpected error occurred.",
            suggestions=[
                "Try refreshing the page",
                "Clear your browser cache and cookies",
                "Try again in a few minutes",
                "Contact support if the problem continues"
            ],
            technical_details=f"Context: {context}, Error: {str(error)}, Traceback: {traceback.format_exc()}",
            error_code="SYSTEM_001"
        )
    
    def display_error(self, error_info: ErrorInfo) -> None:
        """Display error information in Streamlit UI"""
        # Choose appropriate Streamlit display method based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 **Critical Error**\n\n{error_info.user_message}")
        elif error_info.severity == ErrorSeverity.HIGH:
            st.error(f"❌ **Error**\n\n{error_info.user_message}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ **Warning**\n\n{error_info.user_message}")
        else:
            st.info(f"ℹ️ **Notice**\n\n{error_info.user_message}")
        
        # Show suggestions
        if error_info.suggestions:
            st.markdown("**💡 Suggestions:**")
            for suggestion in error_info.suggestions:
                st.markdown(f"• {suggestion}")
        
        # Show technical details in expander for debugging
        if error_info.technical_details:
            with st.expander("🔍 Technical Details (for debugging)"):
                st.code(error_info.technical_details, language="text")
    
    def get_error_history(self) -> List[ErrorInfo]:
        """Get the history of errors"""
        return self.error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear the error history"""
        self.error_history.clear()


class ValidationUtils:
    """Utility functions for input validation"""
    
    @staticmethod
    def validate_csv_file(uploaded_file) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive CSV file validation.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file extension
        if not uploaded_file.name.lower().endswith('.csv'):
            return False, "File must have a .csv extension"
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if uploaded_file.size > max_size:
            return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds the 50MB limit"
        
        # Check if file is empty
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        # Try to read and validate CSV structure
        try:
            uploaded_file.seek(0)
            # Read just the first few rows to validate structure
            sample_df = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)  # Reset for actual processing
            
            # Check if DataFrame is empty
            if sample_df.empty:
                return False, "CSV file contains no data"
            
            # Check for reasonable number of columns
            if sample_df.shape[1] == 0:
                return False, "CSV file has no columns"
            
            if sample_df.shape[1] > 1000:
                return False, f"CSV file has too many columns ({sample_df.shape[1]}). Maximum supported is 1000."
            
            return True, None
            
        except pd.errors.EmptyDataError:
            return False, "CSV file is empty or contains no valid data"
        except pd.errors.ParserError as e:
            return False, f"CSV parsing error: {str(e)}"
        except UnicodeDecodeError:
            return False, "File encoding issue. Please ensure the file is saved as UTF-8"
        except Exception as e:
            return False, f"Unexpected error reading CSV: {str(e)}"
    
    @staticmethod
    def validate_user_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query input.
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Please enter a question about your data"
        
        # Check minimum length
        if len(query.strip()) < 3:
            return False, "Question is too short. Please provide more detail"
        
        # Check maximum length
        if len(query) > 2000:
            return False, "Question is too long. Please keep it under 2000 characters"
        
        # Check for potentially harmful content (basic check)
        harmful_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in harmful_patterns):
            return False, "Question contains potentially harmful content"
        
        return True, None
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate DataFrame for analysis readiness.
        
        Args:
            df: Pandas DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, "No data available"
        
        if df.empty:
            return False, "Dataset is empty"
        
        # Check for reasonable size limits
        if df.shape[0] > 1_000_000:
            return False, f"Dataset is too large ({df.shape[0]:,} rows). Maximum supported is 1,000,000 rows"
        
        if df.shape[1] > 1000:
            return False, f"Dataset has too many columns ({df.shape[1]}). Maximum supported is 1000"
        
        # Check memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        max_memory = 500 * 1024 * 1024  # 500MB
        if memory_usage > max_memory:
            return False, f"Dataset uses too much memory ({memory_usage / 1024 / 1024:.1f}MB). Maximum is 500MB"
        
        return True, None
    
    @staticmethod
    def validate_api_configuration() -> Tuple[bool, Optional[str]]:
        """
        Validate API configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            return False, "GOOGLE_API_KEY environment variable is not set"
        
        if len(api_key.strip()) < 10:
            return False, "GOOGLE_API_KEY appears to be invalid (too short)"
        
        # Basic format check for Google API keys
        if not api_key.startswith(('AIza', 'ya29')):
            return False, "GOOGLE_API_KEY doesn't appear to be a valid Google API key format"
        
        return True, None


class GracefulDegradation:
    """
    Handles graceful degradation when services are unavailable.
    """
    
    @staticmethod
    def check_service_availability() -> Dict[str, bool]:
        """
        Check availability of external services.
        
        Returns:
            Dictionary with service availability status
        """
        services = {
            'google_api': False,
            'internet': False
        }
        
        # Check internet connectivity
        try:
            response = requests.get('https://www.google.com', timeout=5)
            services['internet'] = response.status_code == 200
        except:
            services['internet'] = False
        
        # Check Google API accessibility (basic connectivity test)
        try:
            # Just check if we can reach the Google API endpoint
            response = requests.get('https://generativelanguage.googleapis.com', timeout=5)
            services['google_api'] = True  # If we can reach it, assume it's available
        except:
            services['google_api'] = False
        
        return services
    
    @staticmethod
    def get_offline_capabilities() -> List[str]:
        """
        Get list of capabilities available when offline.
        
        Returns:
            List of available offline features
        """
        return [
            "View uploaded data preview",
            "Basic data statistics (shape, columns, data types)",
            "Export conversation history",
            "File validation and upload",
            "Browse previous conversations"
        ]
    
    @staticmethod
    def display_degraded_mode_info():
        """Display information about degraded mode capabilities"""
        st.warning("⚠️ **Limited Mode**: Some services are unavailable")
        
        st.info("**Available features:**")
        capabilities = GracefulDegradation.get_offline_capabilities()
        for capability in capabilities:
            st.markdown(f"• {capability}")
        
        st.info("**Unavailable features:**")
        st.markdown("• AI-powered data analysis")
        st.markdown("• Natural language queries")
        st.markdown("• Automatic visualization generation")
        
        st.markdown("**💡 To restore full functionality:**")
        st.markdown("• Check your internet connection")
        st.markdown("• Verify your Google API key is configured")
        st.markdown("• Try refreshing the page")


# Global error handler instance
error_handler = ErrorHandler()
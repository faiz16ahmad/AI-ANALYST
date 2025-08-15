# Comprehensive Error Handling System

This document describes the comprehensive error handling system implemented for the CSV Data Analyst application.

## Overview

The error handling system provides:
- **Centralized error management** with structured error information
- **User-friendly error messages** with actionable suggestions
- **Input validation** for files, queries, and data
- **Graceful degradation** when services are unavailable
- **Enhanced error display** in the Streamlit UI

## Components

### 1. ErrorHandler Class

The main error handling class that:
- Classifies errors into categories (API, Network, File, Data, etc.)
- Generates user-friendly messages with suggestions
- Maintains error history for debugging
- Provides structured error information

#### Error Categories
- `API_ERROR`: Issues with Google Gemini API
- `NETWORK_ERROR`: Connectivity problems
- `FILE_ERROR`: CSV file processing issues
- `DATA_ERROR`: DataFrame processing problems
- `VALIDATION_ERROR`: Input validation failures
- `PROCESSING_ERROR`: Agent processing issues
- `CONFIGURATION_ERROR`: Setup and config problems
- `SYSTEM_ERROR`: General system errors

#### Error Severity Levels
- `LOW`: Minor issues that don't prevent operation
- `MEDIUM`: Issues that affect functionality but allow partial operation
- `HIGH`: Serious issues that prevent core functionality
- `CRITICAL`: System-breaking errors

### 2. ValidationUtils Class

Provides comprehensive validation for:

#### CSV File Validation
- File format checking (.csv extension)
- File size limits (50MB maximum)
- Content validation (not empty, parseable)
- Structure validation (reasonable column/row counts)
- Encoding validation (UTF-8)

#### User Query Validation
- Minimum/maximum length checks
- Content safety validation
- Format validation

#### DataFrame Validation
- Size limits (1M rows, 1000 columns max)
- Memory usage limits (500MB max)
- Content validation (not empty)

#### API Configuration Validation
- Environment variable presence
- API key format validation
- Basic key structure checks

### 3. GracefulDegradation Class

Handles service unavailability:

#### Service Availability Checks
- Internet connectivity testing
- Google API accessibility testing
- Service status monitoring

#### Offline Capabilities
When services are unavailable, users can still:
- View uploaded data preview
- Access basic data statistics
- Export conversation history
- Validate and upload files
- Browse previous conversations

#### Degraded Mode Display
- Clear indication of limited functionality
- List of available vs unavailable features
- Recovery instructions

## Integration Points

### Main Application (app.py)
- Service availability checks on startup
- Enhanced file upload validation
- Comprehensive query processing error handling
- Critical error recovery options

### Agent System (csv_analyst_agent.py)
- LLM initialization error handling
- Agent creation error handling
- Query processing error handling
- Enhanced error response generation

### Visualization System (visualization.py)
- Chart generation error handling
- Library-specific error handling
- Graceful fallbacks for visualization failures

### Conversation UI (conversation_ui.py)
- Enhanced error message display
- Error suggestion rendering
- Error category indication

## Error Message Examples

### API Key Issues
```
🔑 API Key Issue: There's a problem with your Google API key.

💡 Suggestions:
• Check that your GOOGLE_API_KEY environment variable is set correctly
• Verify your API key is valid and hasn't expired
• Ensure your API key has access to the Gemini API
• Try regenerating your API key in the Google Cloud Console
```

### Network Issues
```
🌐 Connection Problem: Unable to connect to the AI service.

💡 Suggestions:
• Check your internet connection
• Try refreshing the page
• Verify that your firewall isn't blocking the connection
• If using a VPN, try disconnecting temporarily
```

### File Issues
```
📁 File Problem: There's an issue with your CSV file.

💡 Suggestions:
• Ensure your file is a valid CSV format
• Check that the file isn't corrupted or empty
• Verify the file size is under 50MB
• Try opening the file in a spreadsheet program to check for issues
```

## Usage Examples

### Basic Error Handling
```python
from src.utils.error_handler import error_handler

try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    error_info = error_handler.handle_error(e, "Operation context")
    error_handler.display_error(error_info)
```

### Validation
```python
from src.utils.error_handler import ValidationUtils

# Validate user input
is_valid, error_msg = ValidationUtils.validate_user_query(user_input)
if not is_valid:
    st.error(error_msg)
    return

# Validate DataFrame
is_valid, error_msg = ValidationUtils.validate_dataframe(df)
if not is_valid:
    st.error(error_msg)
    return
```

### Graceful Degradation
```python
from src.utils.error_handler import GracefulDegradation

# Check service availability
services = GracefulDegradation.check_service_availability()
if not services['internet']:
    GracefulDegradation.display_degraded_mode_info()
    return
```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for AI functionality
- `MAX_FILE_SIZE`: Optional, defaults to 50MB
- `MAX_CONVERSATION_HISTORY`: Optional, defaults to 10

### Error Logging
Errors are logged with appropriate levels:
- ERROR: For actual errors that need attention
- DEBUG: For technical details and stack traces
- INFO: For general operational information

## Recovery Mechanisms

### Automatic Recovery
- Service availability re-checking
- Automatic retry for transient errors
- Session state cleanup on critical errors

### User-Initiated Recovery
- Page refresh button
- Session clear button
- Error history viewing
- Manual service re-initialization

## Best Practices

### For Developers
1. Always use the centralized error handler for exceptions
2. Provide meaningful context when handling errors
3. Use appropriate validation before operations
4. Check service availability for network-dependent operations
5. Implement graceful fallbacks where possible

### For Users
1. Check error suggestions before contacting support
2. Verify API key configuration for AI features
3. Ensure stable internet connection
4. Use supported file formats and sizes
5. Try recovery options before restarting

## Testing

Run the error handling tests:
```bash
python test_error_handling.py
```

This will test:
- Error classification and messaging
- Validation utilities
- Service availability checks
- Graceful degradation features

## Future Enhancements

Potential improvements:
- Retry mechanisms with exponential backoff
- More sophisticated service health monitoring
- User preference for error detail levels
- Integration with external monitoring services
- Automated error reporting and analytics
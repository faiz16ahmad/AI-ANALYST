# Configuration management
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import re

# Configure logging for configuration module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration settings with comprehensive validation and management"""
    
    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Application Limits
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "10000"))
    
    # Visualization Settings
    DEFAULT_CHART_LIBRARY: str = os.getenv("DEFAULT_CHART_LIBRARY", "plotly")
    CHART_WIDTH: int = int(os.getenv("CHART_WIDTH", "800"))
    CHART_HEIGHT: int = int(os.getenv("CHART_HEIGHT", "600"))
    
    # File Handling
    SUPPORTED_FILE_TYPES: List[str] = field(default_factory=lambda: [".csv"])
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    
    # Performance Settings
    PANDAS_CHUNKSIZE: int = int(os.getenv("PANDAS_CHUNKSIZE", "10000"))
    MEMORY_LIMIT_MB: int = int(os.getenv("MEMORY_LIMIT_MB", "1000"))
    
    # Security Settings
    ENABLE_CODE_EXECUTION: bool = os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true"
    ALLOWED_FUNCTIONS: List[str] = field(default_factory=lambda: [
        "describe", "info", "head", "tail", "shape", "columns", "dtypes",
        "mean", "median", "std", "min", "max", "count", "sum",
        "groupby", "sort_values", "value_counts", "corr", "plot"
    ])
    
    # Development Settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._setup_logging()
        self._create_directories()
    
    def _setup_logging(self) -> None:
        """Configure logging based on settings"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        if self.DEBUG_MODE:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            Path(self.TEMP_DIR).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {self.TEMP_DIR}")
        except Exception as e:
            logger.warning(f"Could not create directory {self.TEMP_DIR}: {e}")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of configuration settings
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate API key
        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required but not set")
        elif not self._validate_api_key_format(self.GOOGLE_API_KEY):
            errors.append("GOOGLE_API_KEY format appears invalid")
        
        # Validate numeric settings
        if self.MAX_FILE_SIZE <= 0:
            errors.append("MAX_FILE_SIZE must be positive")
        elif self.MAX_FILE_SIZE > 500_000_000:  # 500MB limit
            errors.append("MAX_FILE_SIZE exceeds maximum allowed (500MB)")
        
        if self.MAX_CONVERSATION_HISTORY <= 0:
            errors.append("MAX_CONVERSATION_HISTORY must be positive")
        elif self.MAX_CONVERSATION_HISTORY > 100:
            errors.append("MAX_CONVERSATION_HISTORY exceeds recommended maximum (100)")
        
        if self.MAX_QUERY_LENGTH <= 0:
            errors.append("MAX_QUERY_LENGTH must be positive")
        
        if self.CHART_WIDTH <= 0 or self.CHART_HEIGHT <= 0:
            errors.append("Chart dimensions must be positive")
        
        # Validate chart library
        if self.DEFAULT_CHART_LIBRARY not in ["plotly", "matplotlib"]:
            errors.append("DEFAULT_CHART_LIBRARY must be 'plotly' or 'matplotlib'")
        
        # Validate file types
        if not self.SUPPORTED_FILE_TYPES:
            errors.append("SUPPORTED_FILE_TYPES cannot be empty")
        
        # Validate directories
        if not self._validate_directory_access(self.TEMP_DIR):
            errors.append(f"Cannot access or create TEMP_DIR: {self.TEMP_DIR}")
        
        # Validate memory settings
        if self.MEMORY_LIMIT_MB <= 0:
            errors.append("MEMORY_LIMIT_MB must be positive")
        
        return len(errors) == 0, errors
    
    def _validate_api_key_format(self, api_key: str) -> bool:
        """Validate Google API key format"""
        if not api_key:
            return False
        
        # Google API keys typically start with 'AIza' or 'ya29'
        # and are 39 characters long for AIza keys
        if api_key.startswith('AIza') and len(api_key) == 39:
            return True
        elif api_key.startswith('ya29'):
            return True
        
        return False
    
    def _validate_directory_access(self, directory: str) -> bool:
        """Validate directory can be accessed/created"""
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            return path.exists() and path.is_dir()
        except Exception:
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of configuration
        
        Returns:
            Dict containing health status information
        """
        is_valid, errors = self.validate()
        
        status = {
            "overall_status": "healthy" if is_valid else "unhealthy",
            "validation_errors": errors,
            "api_configured": bool(self.GOOGLE_API_KEY),
            "directories_accessible": self._validate_directory_access(self.TEMP_DIR),
            "memory_settings": {
                "limit_mb": self.MEMORY_LIMIT_MB,
                "chunk_size": self.PANDAS_CHUNKSIZE
            },
            "feature_flags": {
                "code_execution": self.ENABLE_CODE_EXECUTION,
                "debug_mode": self.DEBUG_MODE,
                "profiling": self.ENABLE_PROFILING
            }
        }
        
        return status
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration (excluding sensitive data)"""
        return {
            "api_key_configured": bool(self.GOOGLE_API_KEY),
            "api_key_format_valid": self._validate_api_key_format(self.GOOGLE_API_KEY) if self.GOOGLE_API_KEY else False,
            "max_file_size_mb": self.MAX_FILE_SIZE / (1024 * 1024),
            "max_conversation_history": self.MAX_CONVERSATION_HISTORY,
            "chart_library": self.DEFAULT_CHART_LIBRARY,
            "supported_file_types": self.SUPPORTED_FILE_TYPES,
            "debug_mode": self.DEBUG_MODE,
            "code_execution_enabled": self.ENABLE_CODE_EXECUTION,
            "temp_directory": self.TEMP_DIR
        }
    
    @classmethod
    def create_from_env_file(cls, env_file_path: str) -> 'Config':
        """Create configuration from specific .env file"""
        if Path(env_file_path).exists():
            load_dotenv(env_file_path, override=True)
            logger.info(f"Loaded configuration from {env_file_path}")
        else:
            logger.warning(f"Environment file not found: {env_file_path}")
        
        return cls()
    
    def save_to_env_file(self, env_file_path: str, include_sensitive: bool = False) -> bool:
        """
        Save current configuration to .env file
        
        Args:
            env_file_path: Path to save the .env file
            include_sensitive: Whether to include sensitive data like API keys
            
        Returns:
            bool: Success status
        """
        try:
            with open(env_file_path, 'w') as f:
                f.write("# CSV Data Analyst Configuration\n")
                f.write("# Generated automatically - modify with care\n\n")
                
                if include_sensitive and self.GOOGLE_API_KEY:
                    f.write(f"GOOGLE_API_KEY={self.GOOGLE_API_KEY}\n")
                else:
                    f.write("# GOOGLE_API_KEY=your_api_key_here\n")
                
                f.write(f"\n# Application Limits\n")
                f.write(f"MAX_FILE_SIZE={self.MAX_FILE_SIZE}\n")
                f.write(f"MAX_CONVERSATION_HISTORY={self.MAX_CONVERSATION_HISTORY}\n")
                f.write(f"MAX_QUERY_LENGTH={self.MAX_QUERY_LENGTH}\n")
                f.write(f"MAX_RESPONSE_LENGTH={self.MAX_RESPONSE_LENGTH}\n")
                
                f.write(f"\n# Visualization Settings\n")
                f.write(f"DEFAULT_CHART_LIBRARY={self.DEFAULT_CHART_LIBRARY}\n")
                f.write(f"CHART_WIDTH={self.CHART_WIDTH}\n")
                f.write(f"CHART_HEIGHT={self.CHART_HEIGHT}\n")
                
                f.write(f"\n# Performance Settings\n")
                f.write(f"PANDAS_CHUNKSIZE={self.PANDAS_CHUNKSIZE}\n")
                f.write(f"MEMORY_LIMIT_MB={self.MEMORY_LIMIT_MB}\n")
                
                f.write(f"\n# Development Settings\n")
                f.write(f"DEBUG_MODE={str(self.DEBUG_MODE).lower()}\n")
                f.write(f"LOG_LEVEL={self.LOG_LEVEL}\n")
                f.write(f"ENABLE_PROFILING={str(self.ENABLE_PROFILING).lower()}\n")
                f.write(f"ENABLE_CODE_EXECUTION={str(self.ENABLE_CODE_EXECUTION).lower()}\n")
                
                f.write(f"\n# File Handling\n")
                f.write(f"TEMP_DIR={self.TEMP_DIR}\n")
            
            logger.info(f"Configuration saved to {env_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {env_file_path}: {e}")
            return False


class ConfigurationManager:
    """Manages application configuration with validation and health checks"""
    
    def __init__(self):
        self.config = Config()
        self._startup_validation_performed = False
    
    def perform_startup_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform comprehensive startup validation
        
        Returns:
            Tuple[bool, Dict]: (success, validation_report)
        """
        logger.info("Performing startup validation...")
        
        validation_report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "python_version": sys.version,
            "config_validation": {},
            "environment_check": {},
            "dependency_check": {},
            "health_status": {}
        }
        
        # Validate configuration
        is_config_valid, config_errors = self.config.validate()
        validation_report["config_validation"] = {
            "valid": is_config_valid,
            "errors": config_errors
        }
        
        # Check environment
        env_status = self._check_environment()
        validation_report["environment_check"] = env_status
        
        # Check dependencies
        dep_status = self._check_dependencies()
        validation_report["dependency_check"] = dep_status
        
        # Get health status
        health_status = self.config.get_health_status()
        validation_report["health_status"] = health_status
        
        # Overall success
        overall_success = (
            is_config_valid and 
            env_status["overall_status"] == "ok" and
            dep_status["critical_dependencies_available"]
        )
        
        validation_report["overall_success"] = overall_success
        
        self._startup_validation_performed = True
        
        if overall_success:
            logger.info("✅ Startup validation completed successfully")
        else:
            logger.warning("⚠️ Startup validation completed with issues")
        
        return overall_success, validation_report
    
    def _check_environment(self) -> Dict[str, Any]:
        """Check environment variables and system requirements"""
        env_status = {
            "overall_status": "ok",
            "issues": [],
            "python_version_ok": sys.version_info >= (3, 8),
            "env_file_exists": Path(".env").exists(),
            "required_env_vars": {}
        }
        
        # Check Python version
        if not env_status["python_version_ok"]:
            env_status["issues"].append("Python 3.8+ required")
            env_status["overall_status"] = "error"
        
        # Check .env file
        if not env_status["env_file_exists"]:
            env_status["issues"].append(".env file not found")
            env_status["overall_status"] = "warning"
        
        # Check required environment variables
        required_vars = ["GOOGLE_API_KEY"]
        for var in required_vars:
            value = os.getenv(var)
            env_status["required_env_vars"][var] = {
                "set": bool(value),
                "value_length": len(value) if value else 0
            }
            
            if not value:
                env_status["issues"].append(f"Required environment variable {var} not set")
                env_status["overall_status"] = "error"
        
        return env_status
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check if required dependencies are available"""
        dep_status = {
            "critical_dependencies_available": True,
            "optional_dependencies_available": True,
            "dependencies": {}
        }
        
        # Critical dependencies
        critical_deps = [
            "streamlit", "pandas", "langchain", "langchain_google_genai", 
            "python-dotenv", "matplotlib", "plotly"
        ]
        
        # Optional dependencies
        optional_deps = ["seaborn", "numpy", "scipy"]
        
        for dep in critical_deps:
            try:
                __import__(dep)
                dep_status["dependencies"][dep] = {"available": True, "critical": True}
            except ImportError:
                dep_status["dependencies"][dep] = {"available": False, "critical": True}
                dep_status["critical_dependencies_available"] = False
        
        for dep in optional_deps:
            try:
                __import__(dep)
                dep_status["dependencies"][dep] = {"available": True, "critical": False}
            except ImportError:
                dep_status["dependencies"][dep] = {"available": False, "critical": False}
                dep_status["optional_dependencies_available"] = False
        
        return dep_status
    
    def get_startup_report(self) -> Optional[Dict[str, Any]]:
        """Get the startup validation report if available"""
        if not self._startup_validation_performed:
            return None
        
        return self.config.get_health_status()
    
    def is_healthy(self) -> bool:
        """Check if configuration is healthy"""
        is_valid, _ = self.config.validate()
        return is_valid


# Global configuration manager instance
config_manager = ConfigurationManager()
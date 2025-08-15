"""
Startup validation utilities for CSV Data Analyst application.

This module provides comprehensive validation and health checks for the application
startup process, ensuring all dependencies, configuration, and environment
requirements are met.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


class StartupValidator:
    """Comprehensive startup validation for the CSV Data Analyst application"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent.parent.parent
    
    def run_all_validations(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all startup validations
        
        Returns:
            Tuple[bool, Dict]: (overall_success, detailed_report)
        """
        logger.info("🚀 Starting comprehensive application validation...")
        
        # Clear previous results
        self.validation_results = []
        
        # Run all validation checks
        self._validate_python_environment()
        self._validate_project_structure()
        self._validate_dependencies()
        self._validate_environment_variables()
        self._validate_configuration()
        self._validate_file_permissions()
        self._validate_api_connectivity()
        
        # Generate report
        report = self._generate_validation_report()
        overall_success = all(result.success for result in self.validation_results)
        
        if overall_success:
            logger.info("✅ All validation checks passed!")
        else:
            logger.warning("⚠️ Some validation checks failed. See report for details.")
        
        return overall_success, report
    
    def _validate_python_environment(self) -> None:
        """Validate Python version and environment"""
        logger.info("Validating Python environment...")
        
        # Check Python version
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            self.validation_results.append(ValidationResult(
                success=True,
                message=f"Python version {'.'.join(map(str, current_version))} is supported",
                details={"required": f"{'.'.join(map(str, min_version))}+", "current": f"{'.'.join(map(str, current_version))}"}
            ))
        else:
            self.validation_results.append(ValidationResult(
                success=False,
                message=f"Python version {'.'.join(map(str, current_version))} is not supported",
                details={"required": f"{'.'.join(map(str, min_version))}+", "current": f"{'.'.join(map(str, current_version))}"},
                suggestions=["Upgrade to Python 3.8 or higher"]
            ))
        
        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            self.validation_results.append(ValidationResult(
                success=True,
                message="Running in virtual environment",
                details={"venv_path": sys.prefix}
            ))
        else:
            self.validation_results.append(ValidationResult(
                success=False,
                message="Not running in virtual environment",
                suggestions=[
                    "Create and activate a virtual environment",
                    "Run: python -m venv venv",
                    "Activate: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Unix)"
                ]
            ))
    
    def _validate_project_structure(self) -> None:
        """Validate project directory structure"""
        logger.info("Validating project structure...")
        
        required_dirs = [
            "src",
            "src/agents",
            "src/components",
            "src/utils"
        ]
        
        required_files = [
            "src/__init__.py",
            "src/app.py",
            "src/config.py",
            "src/agents/csv_analyst_agent.py",
            "requirements.txt"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists() or not full_path.is_dir():
                missing_dirs.append(dir_path)
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists() or not full_path.is_file():
                missing_files.append(file_path)
        
        if not missing_dirs and not missing_files:
            self.validation_results.append(ValidationResult(
                success=True,
                message="Project structure is complete",
                details={"checked_dirs": len(required_dirs), "checked_files": len(required_files)}
            ))
        else:
            suggestions = []
            if missing_dirs:
                suggestions.extend([f"Create directory: {d}" for d in missing_dirs])
            if missing_files:
                suggestions.extend([f"Create file: {f}" for f in missing_files])
            
            self.validation_results.append(ValidationResult(
                success=False,
                message="Project structure is incomplete",
                details={"missing_dirs": missing_dirs, "missing_files": missing_files},
                suggestions=suggestions
            ))
    
    def _validate_dependencies(self) -> None:
        """Validate required Python dependencies"""
        logger.info("Validating dependencies...")
        
        critical_deps = [
            "streamlit",
            "pandas",
            "langchain",
            "langchain_google_genai",
            "dotenv",
            "matplotlib",
            "plotly"
        ]
        
        optional_deps = [
            "seaborn",
            "numpy",
            "scipy"
        ]
        
        missing_critical = []
        missing_optional = []
        available_deps = {}
        
        # Check critical dependencies
        for dep in critical_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                available_deps[dep] = {"available": True, "version": version, "critical": True}
            except ImportError:
                missing_critical.append(dep)
                available_deps[dep] = {"available": False, "critical": True}
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                available_deps[dep] = {"available": True, "version": version, "critical": False}
            except ImportError:
                missing_optional.append(dep)
                available_deps[dep] = {"available": False, "critical": False}
        
        if not missing_critical:
            message = "All critical dependencies are available"
            if missing_optional:
                message += f" ({len(missing_optional)} optional dependencies missing)"
            
            self.validation_results.append(ValidationResult(
                success=True,
                message=message,
                details={"dependencies": available_deps}
            ))
        else:
            suggestions = [
                "Install missing dependencies with: pip install -r requirements.txt",
                "Or install individually:"
            ]
            suggestions.extend([f"pip install {dep}" for dep in missing_critical])
            
            self.validation_results.append(ValidationResult(
                success=False,
                message=f"Missing {len(missing_critical)} critical dependencies",
                details={"missing_critical": missing_critical, "missing_optional": missing_optional, "dependencies": available_deps},
                suggestions=suggestions
            ))
    
    def _validate_environment_variables(self) -> None:
        """Validate environment variables and .env file"""
        logger.info("Validating environment variables...")
        
        env_file = self.project_root / ".env"
        env_example_file = self.project_root / ".env.example"
        
        # Check if .env file exists
        env_file_exists = env_file.exists()
        env_example_exists = env_example_file.exists()
        
        required_vars = ["GOOGLE_API_KEY"]
        optional_vars = [
            "MAX_FILE_SIZE", "MAX_CONVERSATION_HISTORY", "DEFAULT_CHART_LIBRARY",
            "DEBUG_MODE", "LOG_LEVEL", "TEMP_DIR"
        ]
        
        env_status = {}
        missing_required = []
        
        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            env_status[var] = {
                "set": bool(value),
                "length": len(value) if value else 0,
                "required": True
            }
            if not value:
                missing_required.append(var)
        
        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            env_status[var] = {
                "set": bool(value),
                "value": value if value else "default",
                "required": False
            }
        
        success = env_file_exists and not missing_required
        
        if success:
            self.validation_results.append(ValidationResult(
                success=True,
                message="Environment configuration is complete",
                details={"env_file_exists": env_file_exists, "variables": env_status}
            ))
        else:
            suggestions = []
            if not env_file_exists:
                if env_example_exists:
                    suggestions.append("Copy .env.example to .env and fill in your values")
                else:
                    suggestions.append("Create a .env file with required environment variables")
            
            if missing_required:
                suggestions.extend([f"Set {var} in your .env file" for var in missing_required])
            
            self.validation_results.append(ValidationResult(
                success=False,
                message="Environment configuration is incomplete",
                details={
                    "env_file_exists": env_file_exists,
                    "missing_required": missing_required,
                    "variables": env_status
                },
                suggestions=suggestions
            ))
    
    def _validate_configuration(self) -> None:
        """Validate application configuration"""
        logger.info("Validating application configuration...")
        
        try:
            # Import and validate config
            sys.path.insert(0, str(self.project_root))
            from src.config import Config, ConfigurationManager
            
            config_manager = ConfigurationManager()
            is_valid, errors = config_manager.config.validate()
            health_status = config_manager.config.get_health_status()
            
            if is_valid:
                self.validation_results.append(ValidationResult(
                    success=True,
                    message="Application configuration is valid",
                    details={"health_status": health_status}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    success=False,
                    message="Application configuration has errors",
                    details={"errors": errors, "health_status": health_status},
                    suggestions=[
                        "Check your .env file for correct values",
                        "Verify API key format and permissions",
                        "Ensure all numeric values are positive"
                    ]
                ))
        
        except Exception as e:
            self.validation_results.append(ValidationResult(
                success=False,
                message=f"Failed to load configuration: {str(e)}",
                suggestions=[
                    "Check that src/config.py exists and is valid",
                    "Verify all imports are available",
                    "Check for syntax errors in configuration files"
                ]
            ))
    
    def _validate_file_permissions(self) -> None:
        """Validate file system permissions"""
        logger.info("Validating file permissions...")
        
        # Check write permissions for temp directory
        temp_dir = self.project_root / "temp"
        
        try:
            temp_dir.mkdir(exist_ok=True)
            test_file = temp_dir / "test_write.tmp"
            
            # Test write
            test_file.write_text("test")
            
            # Test read
            content = test_file.read_text()
            
            # Clean up
            test_file.unlink()
            
            if content == "test":
                self.validation_results.append(ValidationResult(
                    success=True,
                    message="File system permissions are adequate",
                    details={"temp_dir": str(temp_dir), "write_test": "passed", "read_test": "passed"}
                ))
            else:
                raise Exception("Read test failed")
        
        except Exception as e:
            self.validation_results.append(ValidationResult(
                success=False,
                message=f"File system permission issues: {str(e)}",
                details={"temp_dir": str(temp_dir)},
                suggestions=[
                    "Ensure the application has write permissions to the project directory",
                    "Check that the temp directory can be created and written to",
                    "Run the application with appropriate user permissions"
                ]
            ))
    
    def _validate_api_connectivity(self) -> None:
        """Validate API connectivity (basic check)"""
        logger.info("Validating API connectivity...")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            self.validation_results.append(ValidationResult(
                success=False,
                message="Cannot test API connectivity - no API key configured",
                suggestions=["Configure GOOGLE_API_KEY in your .env file"]
            ))
            return
        
        # Basic format validation
        if api_key.startswith('AIza') and len(api_key) == 39:
            self.validation_results.append(ValidationResult(
                success=True,
                message="API key format appears valid",
                details={"key_format": "valid", "key_length": len(api_key)},
                suggestions=["Run the application to test actual API connectivity"]
            ))
        else:
            self.validation_results.append(ValidationResult(
                success=False,
                message="API key format appears invalid",
                details={"key_format": "invalid", "key_length": len(api_key)},
                suggestions=[
                    "Verify your Google API key is correct",
                    "API keys should start with 'AIza' and be 39 characters long",
                    "Generate a new key from Google Cloud Console if needed"
                ]
            ))
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        successful_checks = sum(1 for result in self.validation_results if result.success)
        total_checks = len(self.validation_results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "successful_checks": successful_checks,
                "failed_checks": total_checks - successful_checks,
                "success_rate": (successful_checks / total_checks * 100) if total_checks > 0 else 0,
                "overall_status": "PASS" if successful_checks == total_checks else "FAIL"
            },
            "checks": []
        }
        
        for i, result in enumerate(self.validation_results, 1):
            check_info = {
                "check_number": i,
                "status": "PASS" if result.success else "FAIL",
                "message": result.message,
                "details": result.details,
                "suggestions": result.suggestions
            }
            report["checks"].append(check_info)
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        report_path = self.project_root / filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report saved to: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return ""
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable summary of the validation report"""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("CSV DATA ANALYST - STARTUP VALIDATION REPORT")
        print("="*60)
        print(f"Status: {summary['overall_status']}")
        print(f"Checks: {summary['successful_checks']}/{summary['total_checks']} passed ({summary['success_rate']:.1f}%)")
        print(f"Timestamp: {report['timestamp']}")
        
        if summary['failed_checks'] > 0:
            print(f"\n❌ FAILED CHECKS ({summary['failed_checks']}):")
            for check in report["checks"]:
                if check["status"] == "FAIL":
                    print(f"  • {check['message']}")
                    if check.get("suggestions"):
                        for suggestion in check["suggestions"][:2]:  # Show first 2 suggestions
                            print(f"    → {suggestion}")
        
        print(f"\n✅ PASSED CHECKS ({summary['successful_checks']}):")
        for check in report["checks"]:
            if check["status"] == "PASS":
                print(f"  • {check['message']}")
        
        print("\n" + "="*60)


def main():
    """Main function for running startup validation"""
    validator = StartupValidator()
    success, report = validator.run_all_validations()
    
    # Print summary
    validator.print_summary(report)
    
    # Save detailed report
    report_path = validator.save_report(report)
    if report_path:
        print(f"\nDetailed report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
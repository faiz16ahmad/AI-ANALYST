#!/usr/bin/env python3
"""
Quick health check script for CSV Data Analyst application.

This script provides a fast way to check if the application is properly
configured and ready to run.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

def quick_health_check():
    """Perform a quick health check of the application"""
    print("🏥 CSV Data Analyst - Quick Health Check")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} is too old (3.8+ required)")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is supported")
    
    # Check virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        warnings.append("⚠️ Not running in virtual environment (recommended)")
    
    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print("✅ .env file found")
    else:
        issues.append("❌ .env file not found")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        if api_key.startswith('AIza') and len(api_key) == 39:
            print("✅ Google API key configured and format looks valid")
        else:
            warnings.append("⚠️ Google API key format may be invalid")
    else:
        issues.append("❌ GOOGLE_API_KEY not configured")
    
    # Check critical dependencies
    critical_deps = ["streamlit", "pandas", "langchain", "langchain_google_genai", "matplotlib", "plotly"]
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if not missing_deps:
        print(f"✅ All {len(critical_deps)} critical dependencies available")
    else:
        issues.append(f"❌ Missing dependencies: {', '.join(missing_deps)}")
    
    # Check project structure
    required_files = [
        "src/app.py",
        "src/config.py",
        "src/agents/csv_analyst_agent.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ Project structure is complete")
    else:
        issues.append(f"❌ Missing files: {', '.join(missing_files)}")
    
    # Check configuration
    try:
        from src.config import Config
        config = Config()
        is_valid, errors = config.validate()
        
        if is_valid:
            print("✅ Configuration is valid")
        else:
            issues.append(f"❌ Configuration errors: {'; '.join(errors)}")
    
    except Exception as e:
        issues.append(f"❌ Cannot load configuration: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    
    if not issues:
        print("🎉 HEALTH CHECK PASSED!")
        print("Your application appears to be properly configured.")
        print("\nTo start the application, run:")
        print("streamlit run src/app.py")
        
        if warnings:
            print(f"\n⚠️ {len(warnings)} warnings:")
            for warning in warnings:
                print(f"  {warning}")
    else:
        print(f"❌ HEALTH CHECK FAILED - {len(issues)} issues found:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n🔧 Quick fixes:")
        if "❌ .env file not found" in str(issues):
            print("  • Copy .env.example to .env and configure your API key")
        if "GOOGLE_API_KEY not configured" in str(issues):
            print("  • Add GOOGLE_API_KEY=your_api_key_here to your .env file")
        if "Missing dependencies" in str(issues):
            print("  • Run: pip install -r requirements.txt")
        if "Not running in virtual environment" in str(warnings):
            print("  • Create and activate a virtual environment")
        
        print("\nFor detailed diagnostics, run:")
        print("python src/utils/startup_validator.py")
        
        return False
    
    return True

def main():
    """Main function"""
    try:
        success = quick_health_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed with error: {str(e)}")
        print("\nFor detailed diagnostics, run:")
        print("python src/utils/startup_validator.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
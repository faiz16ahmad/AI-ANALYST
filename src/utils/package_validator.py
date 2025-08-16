"""
Package validation utility for safe code execution
"""

import importlib
import sys
from typing import List, Dict, Set
import re

class PackageValidator:
    """Validates that code only uses available packages"""
    
    # Packages we know are available
    ALLOWED_PACKAGES = {
        'pandas': 'pd',
        'plotly.express': 'px',
        'plotly.graph_objects': 'go', 
        'plotly.subplots': None,
        'numpy': 'np',
        'datetime': None,
        'json': None,
        'math': None,
        're': None
    }
    
    # Packages that are commonly requested but not available
    FORBIDDEN_PACKAGES = {
        'seaborn', 'sns', 'matplotlib', 'plt', 'sklearn', 
        'scipy', 'tensorflow', 'torch', 'bokeh'
    }
    
    @classmethod
    def validate_code(cls, code: str) -> Dict[str, any]:
        """
        Validate that code only uses allowed packages
        
        Returns:
            {
                'valid': bool,
                'issues': List[str],
                'suggestions': List[str]
            }
        """
        issues = []
        suggestions = []
        
        # Extract import statements
        imports = cls._extract_imports(code)
        
        for import_info in imports:
            package = import_info['package']
            alias = import_info['alias']
            
            # Check if package is forbidden
            if package in cls.FORBIDDEN_PACKAGES or alias in cls.FORBIDDEN_PACKAGES:
                issues.append(f"Package '{package}' is not available")
                suggestions.append(cls._suggest_alternative(package))
            
            # Check if package is not in allowed list
            elif package not in cls.ALLOWED_PACKAGES:
                # Try to check if it's actually available
                if not cls._is_package_available(package):
                    issues.append(f"Package '{package}' may not be available")
                    suggestions.append(f"Use one of: {', '.join(cls.ALLOWED_PACKAGES.keys())}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions
        }
    
    @classmethod
    def _extract_imports(cls, code: str) -> List[Dict[str, str]]:
        """Extract import statements from code"""
        imports = []
        
        # Patterns for different import styles
        patterns = [
            r'import\s+(\w+(?:\.\w+)*)\s+as\s+(\w+)',  # import pandas as pd
            r'import\s+(\w+(?:\.\w+)*)',               # import pandas
            r'from\s+(\w+(?:\.\w+)*)\s+import',        # from pandas import DataFrame
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    if 'as' in pattern:
                        package, alias = match.groups()
                        imports.append({'package': package, 'alias': alias})
                    else:
                        package = match.group(1)
                        imports.append({'package': package, 'alias': package.split('.')[-1]})
        
        return imports
    
    @classmethod
    def _is_package_available(cls, package_name: str) -> bool:
        """Check if a package is actually available"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    @classmethod
    def _suggest_alternative(cls, forbidden_package: str) -> str:
        """Suggest alternatives for forbidden packages"""
        alternatives = {
            'seaborn': 'Use plotly.express for statistical plots',
            'sns': 'Use plotly.express for statistical plots', 
            'matplotlib': 'Use plotly.graph_objects for custom plots',
            'plt': 'Use plotly.graph_objects for custom plots',
            'sklearn': 'Use pandas for data manipulation, avoid ML libraries',
            'scipy': 'Use numpy for mathematical operations'
        }
        
        return alternatives.get(forbidden_package, 'Use pandas and plotly instead')


def validate_and_guide_code_generation(query: str, detected_packages: Set[str]) -> str:
    """
    Generate guidance for LLM code generation based on package validation
    """
    forbidden_found = detected_packages.intersection(PackageValidator.FORBIDDEN_PACKAGES)
    
    if forbidden_found:
        guidance = f"""
PACKAGE RESTRICTION DETECTED: The query might require {', '.join(forbidden_found)} which is not available.

ALTERNATIVE APPROACH:
- Use plotly.express for statistical visualizations
- Use plotly.graph_objects for custom charts
- Use pandas for data manipulation
- Avoid seaborn, matplotlib, sklearn

EXAMPLE REPLACEMENTS:
- sns.heatmap() → px.imshow() or go.Heatmap()
- plt.plot() → go.Scatter() or px.line()
- sns.boxplot() → px.box()
"""
        return guidance
    
    return ""
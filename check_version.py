import sys
import importlib.metadata

def check_version():
    """Check the installed EnergyPlus version and display package information."""
    try:
        try:
            version = importlib.metadata.version('pyenergyplus-lbnl')
            package_name = 'pyenergyplus-lbnl'
        except importlib.metadata.PackageNotFoundError:
            version = importlib.metadata.version('pyenergyplus')
            package_name = 'pyenergyplus'
        
        print(f"EnergyPlus Python Library Version: {version}")
        print(f"Package name: {package_name}")
        
        import pyenergyplus
        print(f"Package location: {pyenergyplus.__file__}")
        
        try:
            from pyenergyplus.api import EnergyPlusAPI
            api = EnergyPlusAPI()
            
            print("\nAPI Information:")
            print(f"  - API available: Yes")
            
            try:
                state = api.state_manager.new_state()
                print(f"  - State manager: Working")
                api.state_manager.delete_state(state)
            except Exception as e:
                print(f"  - State manager: Error ({e})")
                
        except Exception as e:
            print(f"\nAPI Error: {e}")
            
    except ImportError:
        print("EnergyPlus library is not installed.")
        print("\nTo install, run:")
        print("  pip install pyenergyplus")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking version: {e}")
        sys.exit(1)


def check_pip_package():
    """Display pip package information."""
    try:
        import subprocess
        result = subprocess.run(
            ['pip', 'show', 'pyenergyplus'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("\n" + "="*60)
            print("Pip Package Information:")
            print("="*60)
            print(result.stdout)
        else:
            print("\nCould not retrieve pip package information.")
    except Exception as e:
        print(f"\nError running pip show: {e}")


if __name__ == "__main__":
    print("="*60)
    print("EnergyPlus Version Checker")
    print("="*60 + "\n")
    
    check_version()
    check_pip_package()
    
    print("\n" + "="*60)
    print("To install a specific version:")
    print("  pip install pyenergyplus==24.1.0")
    print("\nTo upgrade to latest:")
    print("  pip install --upgrade pyenergyplus")
    print("="*60)

import subprocess
import sys
import pkg_resources

def get_conda_packages():
    """Get packages installed via conda"""
    try:
        result = subprocess.run(['conda', 'list'], capture_output=True, text=True, check=True)
        packages = []
        for line in result.stdout.split('\n')[3:]:  # Skip header
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    version = parts[1]
                    # Skip conda itself and python
                    if name not in ['conda', 'python']:
                        packages.append(f"{name}=={version}")
        return packages
    except Exception as e:
        print(f"Error getting conda packages: {e}")
        return []

def get_pip_packages():
    """Get packages installed via pip"""
    try:
        installed_packages = [pkg for pkg in pkg_resources.working_set]
        return [f"{pkg.key}=={pkg.version}" for pkg in installed_packages]
    except Exception as e:
        print(f"Error getting pip packages: {e}")
        return []

def main():
    conda_packages = get_conda_packages()
    pip_packages = get_pip_packages()
    
    # Write to requirements.txt
    with open('requirementstest.txt', 'w') as f:
        f.write("# Packages installed via conda (reference only)\n")
        f.write("# Use environment.yml for exact conda packages\n")
        f.write("# " + "="*50 + "\n")
        for pkg in conda_packages:
            f.write(f"# {pkg}\n")
        
        f.write("\n# Packages to install via pip\n")
        f.write("# " + "="*50 + "\n")
        for pkg in pip_packages:
            # Skip packages that are typically installed via conda
            if any(conda_pkg.split('==')[0].lower() == pkg.split('==')[0].lower() 
                   for conda_pkg in conda_packages):
                continue
            f.write(f"{pkg}\n")
    
    print("requirements.txt generated successfully!")
    print("Note: Use environment.yml for conda packages and requirements.txt for pip packages")

if __name__ == "__main__":
    main()
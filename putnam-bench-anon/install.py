#!/usr/bin/env python3
"""
Quick installation and setup script for Putnam Problem Solver.

This script provides a one-command setup for the entire system.

Usage:
    python install.py                    # Full installation
    python install.py --quick           # Quick setup (minimal dependencies)
    python install.py --check-only      # Just check what would be installed
"""

import asyncio
import sys
import subprocess
import os
from pathlib import Path
import argparse


def print_banner():
    """Print installation banner."""
    print("🚀 Putnam Problem Solver - Quick Install")
    print("=" * 50)
    print("This script will set up everything you need to get started.")
    print()


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_packages(packages: list, check_only: bool = False):
    """Install required packages."""
    if not packages:
        print("✅ All required packages are already installed!")
        return True
    
    print(f"📦 {'Would install' if check_only else 'Installing'}: {', '.join(packages)}")
    
    if check_only:
        return True
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade'
        ] + packages, check=True)
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def check_package_installed(package: str) -> bool:
    """Check if a package is installed."""
    try:
        if package == 'google-generativeai':
            import google.generativeai
        else:
            __import__(package)
        return True
    except ImportError:
        return False


def get_missing_packages(include_optional: bool = True) -> list:
    """Get list of missing packages."""
    # Core packages (always required)
    core_packages = ['openai', 'anthropic', 'google-generativeai', 'psutil']
    
    # Optional packages (for local models)
    optional_packages = ['transformers', 'torch', 'vllm'] if include_optional else []
    
    missing = []
    
    print("🔍 Checking required packages...")
    for package in core_packages:
        if check_package_installed(package):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if include_optional:
        print("\n🔍 Checking optional packages (for local models)...")
        for package in optional_packages:
            if check_package_installed(package):
                print(f"   ✅ {package}")
            else:
                print(f"   ⚠️ {package} (optional)")
                missing.append(package)
    
    return missing


def create_alias():
    """Create putnam command alias."""
    script_dir = Path(__file__).parent.absolute()
    putnam_cli = script_dir / "putnam_cli.py"
    
    # For Unix-like systems
    if os.name != 'nt':
        shell_profile = Path.home() / ".bashrc"
        if not shell_profile.exists():
            shell_profile = Path.home() / ".bash_profile"
        if not shell_profile.exists():
            shell_profile = Path.home() / ".zshrc"
        
        alias_line = f'alias putnam="python {putnam_cli}"'
        
        try:
            # Check if alias already exists
            if shell_profile.exists():
                with open(shell_profile, 'r') as f:
                    content = f.read()
                    if 'alias putnam=' in content:
                        print("✅ 'putnam' alias already exists")
                        return True
            
            # Add alias
            with open(shell_profile, 'a') as f:
                f.write(f"\n# Putnam Problem Solver alias\n{alias_line}\n")
            
            print(f"✅ Added 'putnam' alias to {shell_profile}")
            print(f"💡 Restart your shell or run: source {shell_profile}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not create alias: {e}")
            print(f"💡 You can manually add: {alias_line}")
    
    else:
        # Windows
        print("💡 On Windows, use: python putnam_cli.py <command>")
    
    return False


async def run_setup():
    """Run the configuration setup."""
    print("\n🛠️ Running configuration setup...")
    try:
        from setup_config import ConfigManager
        manager = ConfigManager()
        await manager.interactive_setup()
        return True
    except ImportError:
        print("❌ Configuration setup not available")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Installation completed!")
    print("\n📚 Next Steps:")
    print("   1. Set up API keys: python setup_config.py")
    print("   2. Check health: python health_check.py")
    print("   3. Quick test: python putnam_cli.py test --provider openai")
    print("   4. Solve a problem: python putnam_cli.py solve dataset/1938-A-1.json")
    print("   5. Run benchmark: python benchmark.py --quick-test")
    print("\n💡 Available Scripts:")
    print("   • putnam_cli.py - Main CLI interface")
    print("   • health_check.py - Check provider health")
    print("   • batch_evaluate.py - Batch evaluation")
    print("   • benchmark.py - Performance comparison")
    print("   • setup_config.py - Configuration management")
    print("   • local_models_example.py - Local model examples")
    print("\n📖 Documentation: See README.md for detailed usage")


async def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install Putnam Problem Solver")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick install (core packages only)")
    parser.add_argument("--check-only", action="store_true",
                       help="Check what would be installed without installing")
    parser.add_argument("--no-setup", action="store_true",
                       help="Skip interactive configuration setup")
    parser.add_argument("--no-alias", action="store_true",
                       help="Skip creating command alias")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python():
        return 1
    
    # Check packages
    missing_packages = get_missing_packages(include_optional=not args.quick)
    
    if args.check_only:
        if missing_packages:
            print(f"\n📋 Would install: {', '.join(missing_packages)}")
        else:
            print("\n✅ All packages are already installed!")
        return 0
    
    # Install packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} packages...")
        if not install_packages(missing_packages):
            return 1
    else:
        print("\n✅ All packages are already installed!")
    
    # Create alias
    if not args.no_alias:
        print("\n🔗 Creating command alias...")
        create_alias()
    
    # Run configuration setup
    if not args.no_setup:
        if input("\n🛠️ Run configuration setup now? (y/n): ").lower().startswith('y'):
            await run_setup()
        else:
            print("💡 You can run setup later with: python setup_config.py")
    
    # Print next steps
    print_next_steps()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main())) 
#!/usr/bin/env python3
"""
Configuration setup script for Putnam Problem Solver.

This script helps users set up API keys, configure providers,
and verify their environment for mathematical problem solving.

Usage:
    python setup_config.py                      # Interactive setup
    python setup_config.py --check              # Check current configuration  
    python setup_config.py --provider openai    # Setup specific provider
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Any, Optional
import getpass
import subprocess

# Add the loader module to the path
sys.path.append(str(Path(__file__).parent))

from loader import get_supported_providers, get_default_models


class ConfigManager:
    """Configuration manager for Putnam Problem Solver."""
    
    def __init__(self):
        self.config_file = Path.home() / ".putnam_config.json"
        self.env_file = Path.home() / ".putnam_env"
    
    def print_banner(self):
        """Print setup banner."""
        print("🛠️ Putnam Problem Solver Configuration Setup")
        print("=" * 55)
        print("This script will help you configure API keys and settings.")
        print()
    
    def load_config(self) -> Dict[str, Any]:
        """Load existing configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {self.config_file}")
    
    def update_env_file(self, env_vars: Dict[str, str]):
        """Update environment file."""
        lines = []
        
        # Read existing lines
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Remove existing vars that we're updating
        lines = [line for line in lines if not any(line.startswith(f"{var}=") for var in env_vars)]
        
        # Add new vars
        for var, value in env_vars.items():
            if value:
                lines.append(f"export {var}={value}")
        
        # Write back
        with open(self.env_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Putnam Problem Solver Environment Variables\n")
            f.write("# Source this file: source ~/.putnam_env\n\n")
            for line in lines:
                f.write(line + "\n")
        
        print(f"✅ Environment file updated: {self.env_file}")
        print(f"💡 Add to your shell profile: source {self.env_file}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies."""
        dependencies = {
            'python': True,  # We're running Python
            'pip': self._command_exists('pip'),
            'openai': self._package_installed('openai'),
            'anthropic': self._package_installed('anthropic'),
            'google-generativeai': self._package_installed('google-generativeai'),
            'transformers': self._package_installed('transformers'),
            'torch': self._package_installed('torch'),
            'vllm': self._package_installed('vllm'),
            'psutil': self._package_installed('psutil')
        }
        return dependencies
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists."""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _package_installed(self, package: str) -> bool:
        """Check if a Python package is installed."""
        try:
            if package == 'google-generativeai':
                import google.generativeai
            else:
                __import__(package)
            return True
        except ImportError:
            return False
    
    def install_dependencies(self, packages: list):
        """Install missing dependencies."""
        if not packages:
            print("✅ All dependencies are installed!")
            return
        
        print(f"📦 Installing missing packages: {', '.join(packages)}")
        
        # Create requirements for missing packages
        package_map = {
            'openai': 'openai',
            'anthropic': 'anthropic',
            'google-generativeai': 'google-generativeai',
            'transformers': 'transformers',
            'torch': 'torch',
            'vllm': 'vllm',
            'psutil': 'psutil'
        }
        
        to_install = [package_map[pkg] for pkg in packages if pkg in package_map]
        
        if to_install:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + to_install, 
                             check=True)
                print("✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
    
    def setup_provider(self, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup a specific provider."""
        print(f"\n🔧 Setting up {provider.upper()}")
        print("-" * 30)
        
        provider_config = config.get('providers', {}).get(provider, {})
        
        if provider == 'openai':
            api_key = self._get_api_key(
                "OpenAI API Key", 
                provider_config.get('api_key'),
                "Get your key from: https://platform.openai.com/api-keys"
            )
            if api_key:
                provider_config['api_key'] = api_key
                os.environ['OPENAI_API_KEY'] = api_key
        
        elif provider == 'anthropic':
            api_key = self._get_api_key(
                "Anthropic API Key",
                provider_config.get('api_key'),
                "Get your key from: https://console.anthropic.com/"
            )
            if api_key:
                provider_config['api_key'] = api_key
                os.environ['ANTHROPIC_API_KEY'] = api_key
        
        elif provider == 'gemini':
            api_key = self._get_api_key(
                "Google API Key",
                provider_config.get('api_key'), 
                "Get your key from: https://makersuite.google.com/app/apikey"
            )
            if api_key:
                provider_config['api_key'] = api_key
                os.environ['GOOGLE_API_KEY'] = api_key
                
        elif provider == 'kimi':
            api_key = self._get_api_key(
                "Kimi/Moonshot API Key",
                provider_config.get('api_key'),
                "Get your key from: https://platform.moonshot.ai/"
            )
            if api_key:
                provider_config['api_key'] = api_key
                os.environ['MOONSHOT_API_KEY'] = api_key
        
        elif provider == 'vllm':
            current_url = provider_config.get('base_url', 'http://localhost:8000/v1')
            print(f"Current VLLM server URL: {current_url}")
            new_url = input("Enter VLLM server URL (press Enter to keep current): ").strip()
            if new_url:
                provider_config['base_url'] = new_url
            else:
                provider_config['base_url'] = current_url
            
            print("💡 Make sure your VLLM server is running:")
            print("   vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8000")
        
        elif provider == 'huggingface':
            print("HuggingFace runs locally - no API key needed.")
            device = provider_config.get('device', 'auto')
            print(f"Current device setting: {device}")
            new_device = input("Device (auto/cuda/cpu) [press Enter to keep current]: ").strip()
            if new_device in ['auto', 'cuda', 'cpu']:
                provider_config['device'] = new_device
            
            print("💡 HuggingFace will download models on first use.")
        
        # Update config
        if 'providers' not in config:
            config['providers'] = {}
        config['providers'][provider] = provider_config
        
        return config
    
    def _get_api_key(self, prompt: str, current_key: Optional[str], help_text: str) -> Optional[str]:
        """Get API key from user."""
        if current_key:
            masked_key = current_key[:8] + "..." if len(current_key) > 8 else "***"
            print(f"Current key: {masked_key}")
            
            if input("Update API key? (y/n): ").lower().startswith('y'):
                print(help_text)
                return getpass.getpass(f"Enter {prompt}: ").strip()
            else:
                return current_key
        else:
            print(f"No {prompt} found.")
            print(help_text)
            if input("Enter API key now? (y/n): ").lower().startswith('y'):
                return getpass.getpass(f"Enter {prompt}: ").strip()
        
        return None
    
    async def test_provider(self, provider: str) -> bool:
        """Test a provider configuration."""
        print(f"🧪 Testing {provider}...")
        
        try:
            from loader import create_loader
            
            loader_kwargs = {}
            if provider == 'vllm':
                config = self.load_config()
                vllm_config = config.get('providers', {}).get('vllm', {})
                loader_kwargs['base_url'] = vllm_config.get('base_url', 'http://localhost:8000/v1')
            elif provider == 'huggingface':
                config = self.load_config()
                hf_config = config.get('providers', {}).get('huggingface', {})
                loader_kwargs['device'] = hf_config.get('device', 'cpu')
                loader_kwargs['solver_model'] = 'microsoft/DialoGPT-small'
                loader_kwargs['grader_model'] = 'microsoft/DialoGPT-small'
            
            loader = create_loader(provider, **loader_kwargs)
            
            # Simple health check
            is_healthy = await loader.health_check()
            
            if is_healthy:
                print(f"✅ {provider} is working correctly!")
                return True
            else:
                print(f"❌ {provider} health check failed")
                return False
                
        except Exception as e:
            print(f"❌ {provider} test failed: {str(e)}")
            return False
    
    def check_current_config(self):
        """Check and display current configuration."""
        print("📋 Current Configuration Status")
        print("=" * 40)
        
        # Environment variables
        print("\n🔧 Environment Variables:")
        env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
        }
        
        for var, value in env_vars.items():
            if value:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                print(f"   ✅ {var}: {masked}")
            else:
                print(f"   ❌ {var}: Not set")
        
        # Dependencies
        print("\n📦 Dependencies:")
        deps = self.check_dependencies()
        for dep, installed in deps.items():
            status = "✅" if installed else "❌"
            print(f"   {status} {dep}")
        
        # Config file
        print(f"\n📄 Config file: {self.config_file}")
        if self.config_file.exists():
            print("   ✅ Exists")
            config = self.load_config()
            providers = config.get('providers', {})
            if providers:
                print("   Configured providers:")
                for provider in providers:
                    print(f"     • {provider}")
        else:
            print("   ❌ Not found")
        
        # Environment file
        print(f"\n🌍 Environment file: {self.env_file}")
        if self.env_file.exists():
            print("   ✅ Exists")
            print(f"   💡 Source with: source {self.env_file}")
        else:
            print("   ❌ Not found")
    
    async def interactive_setup(self):
        """Run interactive setup."""
        self.print_banner()
        
        # Check dependencies first
        print("🔍 Checking dependencies...")
        deps = self.check_dependencies()
        missing_deps = [dep for dep, installed in deps.items() if not installed and dep != 'pip']
        
        if missing_deps:
            print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
            if input("Install missing dependencies? (y/n): ").lower().startswith('y'):
                self.install_dependencies(missing_deps)
        
        # Load existing config
        config = self.load_config()
        
        # Provider setup
        print(f"\n🤖 Available providers: {', '.join(get_supported_providers())}")
        
        # Ask which providers to configure
        if input("\nConfigure all providers? (y/n): ").lower().startswith('y'):
            providers_to_setup = get_supported_providers()
        else:
            providers_to_setup = []
            for provider in get_supported_providers():
                if input(f"Configure {provider}? (y/n): ").lower().startswith('y'):
                    providers_to_setup.append(provider)
        
        # Setup each provider
        env_vars = {}
        for provider in providers_to_setup:
            config = self.setup_provider(provider, config)
            
            # Collect env vars
            provider_config = config.get('providers', {}).get(provider, {})
            if provider == 'openai' and 'api_key' in provider_config:
                env_vars['OPENAI_API_KEY'] = provider_config['api_key']
            elif provider == 'anthropic' and 'api_key' in provider_config:
                env_vars['ANTHROPIC_API_KEY'] = provider_config['api_key']
            elif provider == 'gemini' and 'api_key' in provider_config:
                env_vars['GOOGLE_API_KEY'] = provider_config['api_key']
        
        # Save configuration
        self.save_config(config)
        
        # Update environment file
        if env_vars:
            self.update_env_file(env_vars)
        
        # Test providers
        if input("\nTest configured providers? (y/n): ").lower().startswith('y'):
            print("\n🧪 Testing providers...")
            for provider in providers_to_setup:
                await self.test_provider(provider)
        
        print("\n🎉 Setup completed!")
        print("\n💡 Next steps:")
        print("   1. Source environment file: source ~/.putnam_env")
        print("   2. Test a provider: python putnam_cli.py test --provider openai")
        print("   3. Solve a problem: python putnam_cli.py solve dataset/1938-A-1.json")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Configure Putnam Problem Solver")
    parser.add_argument("--check", action="store_true", help="Check current configuration")
    parser.add_argument("--provider", choices=get_supported_providers(),
                       help="Setup specific provider only")
    parser.add_argument("--install-deps", action="store_true", help="Install missing dependencies")
    parser.add_argument("--test", choices=get_supported_providers(),
                       help="Test specific provider")
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    try:
        if args.check:
            manager.check_current_config()
        elif args.install_deps:
            deps = manager.check_dependencies()
            missing = [dep for dep, installed in deps.items() if not installed and dep != 'pip']
            manager.install_dependencies(missing)
        elif args.test:
            await manager.test_provider(args.test)
        elif args.provider:
            manager.print_banner()
            config = manager.load_config()
            config = manager.setup_provider(args.provider, config)
            manager.save_config(config)
            
            # Test the provider
            if input(f"Test {args.provider}? (y/n): ").lower().startswith('y'):
                await manager.test_provider(args.provider)
        else:
            # Interactive setup
            await manager.interactive_setup()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main())) 
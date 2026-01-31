import os
import subprocess
import sys

def run_command(command):
    return subprocess.run(command, shell=True, check=True)

def setup():
    print("🛡️ Talos MD5: Initializing Setup...")
    
    # 1. Check Python Version
    if sys.version_info[:2] != (3, 11):
        print("❌ Error: Please use Python 3.11 for Talos MD5.")
        return

    # 2. Create Virtual Environment
    if not os.path.exists(".venv"):
        print("📦 Creating Virtual Environment (.venv)...")
        run_command("python -m venv .venv")

    # 3. Path to VENV Python
    if os.name == 'nt': # Windows
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
        venv_pip = os.path.join(".venv", "Scripts", "pip.exe")
    else:
        venv_python = os.path.join(".venv", "bin", "python")
        venv_pip = os.path.join(".venv", "bin", "pip")

    # 4. Install Requirements
    if os.path.exists("requirements.txt"):
        print("📥 Installing Dependencies...")
        run_command(f"{venv_pip} install -r requirements.txt")
    else:
        print("⚠️ requirements.txt not found! Installing core libraries...")
        run_command(f"{venv_pip} install customtkinter darkdetect scikit-learn pandas numpy")

    print("\n✅ Setup Complete! Launching Talos MD5...")
    
    # 5. Run the Main App
    run_command(f"{venv_python} talos.py")

if __name__ == "__main__":
    setup()
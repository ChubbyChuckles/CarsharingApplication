# Project Template

[![CI](https://github.com/ChubbyChuckles/project-template/actions/workflows/ci.yml/badge.svg)](https://github.com/ChubbyChuckles/project-template/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/project-template/badge/?version=latest)](https://project-template.readthedocs.io)

This is a Python project template with automated setup for creating new projects, including a virtual environment, dependency installation, Sphinx documentation, and Git workflow with pre-commit checks.

## Setup Instructions

This template automates the setup of a new Python project. Follow these steps to initialize a new project after cloning or using this template.

### Prerequisites (Outside VS Code)

Before starting, ensure the following are set up on your Windows 10 system:

1. **Install Git**:

   - Download and install Git for Windows from [https://git-scm.com/](https://git-scm.com/).
   - Verify installation by running in Command Prompt or PowerShell:
     ```powershell
     git --version
     ```
   - Ensure Git is added to your system PATH (selected during installation).

2. **Install Python**:

   - Ensure Python 3.12.3 or later is installed. Download from [https://www.python.org/](https://www.python.org/).
   - Verify installation:
     ```powershell
     python --version
     ```
   - Ensure `pip` is available:
     ```powershell
     python -m pip --version
     ```

3. **Set PowerShell Execution Policy**:

   - To run PowerShell scripts like `scripts/commit-push.ps1`, set the execution policy to allow scripts:
     ```powershell
     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
     ```
   - Run this in an elevated PowerShell prompt (right-click PowerShell and select "Run as administrator").
   - If prompted, type `Y` to confirm.

4. **Install Visual Studio Code (Optional but Recommended)**:
   - Download and install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/).
   - Install the Python extension for VS Code (by Microsoft) for better Python support:
     - Open VS Code, go to the Extensions view (`Ctrl+Shift+X`), search for "Python," and install the Microsoft Python extension.

### Setup Steps

1. **Clone the Repository or Create a New Project**:

   - Clone this repository:
     ```powershell
     git clone https://github.com/ChubbyChuckles/project-template.git <new-project-name>
     cd <new-project-name>
     ```
   - Alternatively, use the "Use this template" button on GitHub to create a new repository, then clone it:
     ```powershell
     git clone https://github.com/<your-username>/<your-new-repo>.git
     cd <your-new-repo>
     ```

2. **Run the Bootstrap Script**:

   - Run the `bootstrap.py` script to automate setup:
     ```powershell
     python bootstrap.py
     ```
   - Follow the prompts:
     - **Enter the new project name** (e.g., `MyNewProject`).
     - **Enter the new GitHub repository URL** (e.g., `https://github.com/ChubbyChuckles/my-new-project.git`).
   - The script will:
     - Create a virtual environment (`.venv`).
     - Install dependencies from `requirements.txt` (e.g., `numpy`, `pandas`, `matplotlib`, `sphinx`, `pre-commit`).
     - Create a `.env` file in the root directory with the project name and placeholder environment variables.
     - Update `README.md`, `setup.py`, and `docs/source/conf.py` with the new project name and author.
     - Initialize a Git repository (if needed) and set the new remote URL.
     - Create and switch to a `develop` branch.
     - Run `scripts/commit-push.ps1` to stage, commit, and push changes to the `develop` branch.

3. **Inside VS Code**:
   - **Open the Project**:
     - Launch VS Code and open the project folder:
       ```powershell
       code .
       ```
     - Alternatively, open VS Code, go to `File > Open Folder`, and select the project directory (e.g., `C:\Users\Chuck\Desktop\CR_AI_Engineering\Projekte\Github_Repo_Template\<new-project-name>`).
   - **Select Python Interpreter**:
     - Press `Ctrl+Shift+P` to open the Command Palette.
     - Type `Python: Select Interpreter` and select the virtual environment (`.venv\Scripts\python.exe`).
   - **Run `bootstrap.py` in VS Code** (if not run earlier):
     - Open `bootstrap.py` in VS Code.
     - Right-click the file and select `Run Python File in Terminal`, or use the integrated terminal:
       ```powershell
       python bootstrap.py
       ```
   - **Edit Configuration Files**:
     - Update `docs/source/conf.py` for additional Sphinx settings (e.g., add custom modules for `autodoc`).
     - Modify `README.md` or `setup.py` to add project-specific details.
     - Use VS Code’s built-in Git integration (Source Control tab) to stage, commit, and push changes:
       - Click the Source Control icon in the sidebar.
       - Stage changes by clicking the `+` next to modified files.
       - Enter a commit message and click the checkmark to commit.
       - Click the `...` menu and select `Push` to push to the remote repository.
   - **Generate Sphinx Documentation**:
     - Open the integrated terminal in VS Code (`Ctrl+``).
     - Run:
       ```powershell
       cd docs
       .\make.bat html
       ```
     - Open `docs/build/html/index.html` in a browser to verify the documentation.

### Troubleshooting

- **Pre-Commit Hook Failures**:

  - If `scripts/commit-push.ps1` fails due to pre-commit hooks (e.g., `black`, `flake8`, `sphinx-build`), check the error output in the terminal.
  - Ensure all dependencies are installed:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
  - Verify Sphinx files (`docs/Makefile`, `docs/make.bat`, `docs/source/conf.py`, `docs/source/index.rst`) exist.
  - If issues persist, open `.pre-commit-config.yaml` in VS Code and check the `sphinx-build` hook configuration:
    ```yaml
    - repo: local
      hooks:
        - id: sphinx-build
          name: Build Sphinx documentation
          entry: make html
          language: system
          files: ^docs/
    ```

- **Git Push Errors**:

  - If `git push` fails, verify the GitHub URL and ensure you have push access.
  - Use a personal access token if authentication fails:
    ```powershell
    git remote set-url origin https://<username>:<token>@github.com/<username>/<repo>.git
    ```
    Generate a token in GitHub: `Settings > Developer settings > Personal access tokens > Tokens (classic)`.

- **Dependency Issues**:

  - If `pip install -r requirements.txt` fails, check for conflicting versions in `requirements.txt`.
  - Run in the VS Code terminal:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

- **Virtual Environment Issues**:

  - Ensure `.venv` is not ignored in `.gitignore`.
  - If the virtual environment fails to activate, recreate it:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

- **.env File Issues**:
  - The `bootstrap.py` script creates a `.env` file with the project name and example variables.
  - Edit `.env` in VS Code to add project-specific environment variables (e.g., API keys, database URLs).
  - Do not commit sensitive data to `.env`. For production use, add `.env` to `.gitignore` after setup and use a `.env.example` file for templates.

### Documentation

- Documentation is built with Sphinx. To generate HTML documentation:
  ```powershell
  cd docs
  .\make.bat html
  ```

## Repository Structure

- `bootstrap.py`: Automates project setup.
- `requirements.txt`: Lists dependencies (e.g., `numpy`, `pandas`, `sphinx`, `pre-commit`).
- `scripts/commit-push.ps1`: PowerShell script for staging, committing, and pushing changes.
- `docs/`: Contains Sphinx files (`Makefile`, `make.bat`, `source/conf.py`, `source/index.rst`).
- `.pre-commit-config.yaml`: Configures pre-commit hooks.
- `setup.py`: Python package configuration.
- `.env`: Template file for environment variables.
- `README.md`: This file.

For further customization, edit `setup.py`, `docs/source/conf.py`, `.env`, or add Python modules to the repository.

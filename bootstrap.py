import getpass
import os
import subprocess
import sys
import venv


def run_command(
    command: str, shell: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=shell, check=check, text=True, capture_output=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e.stderr}")
        sys.exit(1)


def main() -> None:
    """Entry point for the project setup script."""
    # Prompt for project name and GitHub URL
    project_name: str = input("Enter the new project name: ").strip()
    github_url: str = input(
        "Enter the new GitHub repository URL (e.g., https://github.com/username/repo.git): "
    ).strip()

    # Validate inputs
    if not project_name or not github_url:
        print("Error: Project name and GitHub URL cannot be empty.")
        sys.exit(1)

    # Step 1: Create virtual environment
    venv_dir: str = ".venv"
    print(f"Creating virtual environment in {venv_dir}...")
    venv.create(venv_dir, with_pip=True)

    # Activate virtual environment
    if sys.platform == "win32":
        activate_script: str = os.path.join(venv_dir, "Scripts", "activate.bat")
    else:
        activate_script: str = os.path.join(venv_dir, "bin", "activate")

    # Step 2: Install dependencies from requirements.txt
    print("Installing dependencies from requirements.txt...")
    if sys.platform == "win32":
        pip_cmd: str = f"{venv_dir}\\Scripts\\pip.exe install -r requirements.txt"
    else:
        pip_cmd: str = f". {activate_script} && pip install -r requirements.txt"
    run_command(pip_cmd)

    # Create Documentation
    print("Generating Sphinx documentation...")
    if sys.platform == "win32":
        run_command("cd docs && make.bat html")
    else:
        run_command("cd docs && make html")

    # Step 3: Create .env file
    env_file_path: str = ".env"
    if not os.path.exists(env_file_path):
        print(f"Creating {env_file_path}...")
        with open(env_file_path, "w") as f:
            f.write(
                "# Environment variables for the project\n"
                f"PROJECT_NAME={project_name}\n"
                "# Add other environment variables here\n"
                "# EXAMPLE_API_KEY=your_api_key_here\n"
                "# DATABASE_URL=your_database_url_here\n"
            )
        print(f"Created {env_file_path} with default configuration")

    # Step 4: Update project name in key files (optional, e.g., README.md, conf.py)
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            readme_content: str = f.read()
        readme_content = readme_content.replace("project-template", project_name)
        with open("README.md", "w") as f:
            f.write(readme_content)
        print(f"Updated project name in README.md to '{project_name}'")

    if os.path.exists("docs/source/conf.py"):
        with open("docs/source/conf.py", "r") as f:
            conf_content: str = f.read()
        conf_content = conf_content.replace(
            "project = 'project-template'", f"project = '{project_name}'"
        )
        conf_content = conf_content.replace(
            "author = 'ChubbyChuckles'", f"author = '{getpass.getuser()}'"
        )
        with open("docs/source/conf.py", "w") as f:
            f.write(conf_content)
        print("Updated project name and author in docs/source/conf.py")

    # Add to main() after updating conf.py
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            pyproject_content: str = f.read()
        pyproject_content = pyproject_content.replace(
            'name = "project-template"', f'name = "{project_name}"'
        )
        pyproject_content = pyproject_content.replace(
            'Homepage = "https://github.com/ChubbyChuckles/project-template"',
            f'Homepage = "{github_url}"',
        )
        pyproject_content = pyproject_content.replace(
            'authors = [{name = "ChubbyChuckles", email = "christian.rickert.1989@gmail.com"}]',
            f'authors = [{{name = "{getpass.getuser()}", email = ""}}]',
        )
        with open("pyproject.toml", "w") as f:
            f.write(pyproject_content)
        print("Updated project name, URL, and author in pyproject.toml")

    if os.path.exists("LICENSE"):
        with open("LICENSE", "r") as f:
            license_content: str = f.read()
        license_content = license_content.replace("ChubbyChuckles", getpass.getuser())
        with open("LICENSE", "w") as f:
            f.write(license_content)
        print("Updated copyright holder in LICENSE")

    # Step 5: Initialize Git (in case .git was removed for template)
    if not os.path.exists(".git"):
        print("Initializing new Git repository...")
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit for new project"')

    # Step 6: Set up remote repository
    print(f"Setting up remote repository: {github_url}")
    run_command("git remote remove origin", check=False)  # Remove existing origin if any
    run_command(f"git remote add origin {github_url}")

    # Step 7: Create and switch to develop branch
    print("Creating and switching to develop branch...")
    run_command("git checkout -b develop")

    # Step 8: Run commit-push.ps1
    print("Running commit-push.ps1...")
    if sys.platform == "win32":
        run_command("powershell -File scripts\\commit-push.ps1")
    else:
        print("Error: commit-push.ps1 is Windows-specific. Please adapt for non-Windows systems.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Contributing to Categorical-Trading

Thank you for considering contributing to the Categorical-Trading project, a Python template repository designed to streamline project setup with automated linting, testing, and documentation. We value contributions that enhance functionality, improve code quality, or expand documentation. This guide outlines the steps to contribute effectively and maintain the project's high standards.

## Contribution Workflow

1. **Fork and Clone the Repository**:

   - Fork the repository on GitHub to your account.
   - Clone your fork locally:
     ```bash
     git clone https://github.com/<your-username>/Categorical-Trading.git
     cd Categorical-Trading
     ```
   - Create a feature branch from the `develop` branch:
     ```bash
     git checkout develop
     git checkout -b feature/<your-feature-name>
     ```

2. **Set Up the Development Environment**:

   - Follow the setup instructions in `README.md` to create a virtual environment and install dependencies.
   - Install pre-commit hooks to enforce code quality:
     ```bash
     pre-commit install
     ```

3. **Develop and Test Your Changes**:

   - Write code in `src/` or tests in `tests/`.
   - Ensure all tests pass using `pytest`:
     ```bash
     pytest tests/
     ```
   - Add or update tests for new features or bug fixes in `tests/`.
   - Update Sphinx documentation in `docs/source/` and build it to verify:
     ```bash
     cd docs
     make html
     ```

4. **Run Pre-commit Checks**:

   - Run all pre-commit hooks to ensure compliance with linting and formatting standards:
     ```bash
     pre-commit run --all-files
     ```
   - Address any failures (e.g., `pylint`, `mypy`, `black`) by reviewing error messages or `pre-commit.log`.

5. **Commit and Push Changes**:

   - Use the provided `commit-push.ps1` script (Windows) to automate formatting, linting, and committing:
     ```powershell
     .\scripts\commit-push.ps1
     ```
   - For non-Windows systems, adapt the script or manually run:
     ```bash
     git add .
     pre-commit run --all-files
     git commit -m "Your commit message"
     git push origin feature/<your-feature-name>
     ```

6. **Submit a Pull Request**:
   - Create a pull request (PR) from your feature branch to the `develop` branch of the main repository (`https://github.com/ChubbyChuckles/project-template`).
   - Provide a clear description of your changes, including motivation, implementation details, and any relevant issues addressed.
   - Ensure the CI workflow (configured in `.github/workflows/ci.yml`) passes, as it runs pre-commit checks and tests.

## Code Quality Standards

To ensure the Categorical-Trading project maintains a robust, maintainable, and scalable codebase, all contributions must adhere to the following code quality standards. These guidelines leverage automated tools and best practices to enforce consistency, readability, and correctness. Compliance is verified through pre-commit hooks and CI workflows, as configured in `.pre-commit-config.yaml` and `.github/workflows/ci.yml`.

### 1. Code Style and Formatting

- **Adopt Black Formatting**:

  - All Python code must conform to the `black` code formatter (version specified in `requirements.txt`) to ensure consistent style.
  - Run `black` manually to format your code:
    ```bash
    black src/ tests/
    ```
  - Pre-commit hooks automatically enforce `black` formatting on staged files.

- **Line Endings and Git Configuration**:

  - Use LF (Unix-style) line endings for all files to ensure cross-platform compatibility.
  - Configure Git to handle line endings correctly:
    ```bash
    git config core.autocrlf false
    git config core.eol lf
    ```
  - The `check-eol` pre-commit hook verifies LF line endings.

- **File Structure**:
  - Place source code in `src/`, tests in `tests/`, and documentation in `docs/source/`.
  - Follow the project's modular structure, organizing code into logically grouped modules (e.g., `src/categorical_trading/utils/` for utilities).

### 2. Linting and Static Analysis

- **Pylint Compliance**:

  - All Python code must pass `pylint` checks (version specified in `requirements.txt`) with a minimum score of 8/10 per module.
  - Address `pylint` errors and warnings, particularly those related to code smells (e.g., `W0611`, `W0703`) or fatal errors (e.g., `F0401`).
  - Review `.pylintrc` for project-specific configurations, such as disabled checks or custom naming conventions.
  - Run `pylint` manually to verify:
    ```bash
    pylint src/ tests/
    ```

- **Type Checking with Mypy**:
  - Use static type checking with `mypy` to enforce type safety and catch type-related errors early.
  - Annotate all functions and classes with type hints, following PEP 484 and PEP 526.
  - Ensure `mypy` checks pass with strict settings (see `mypy.ini` for configuration):
    ```bash
    mypy src/ tests/
    ```
  - Pre-commit hooks enforce `mypy` compliance.

### 3. Testing Standards

- **Unit and Integration Tests**:

  - Write comprehensive tests for all new features and bug fixes using `pytest` (version specified in `requirements.txt`).
  - Place tests in `tests/` with a clear naming convention (e.g., `test_<module>.py`).
  - Aim for at least 85% code coverage, verifiable with:
    ```bash
    pytest --cov=src/ tests/
    ```
  - Tests must cover edge cases, error conditions, and typical use cases.

- **Test Organization**:

  - Mirror the `src/` directory structure in `tests/` (e.g., `tests/utils/test_utils.py` for `src/categorical_trading/utils/utils.py`).
  - Use descriptive test names (e.g., `test_function_handles_invalid_input_raises_valueerror`).
  - Include docstrings for test functions to explain their purpose.

- **Test Execution**:
  - Ensure all tests pass before submitting a pull request:
    ```bash
    pytest tests/
    ```
  - The CI workflow runs `pytest` automatically to validate contributions.

### 4. Documentation Standards

- **Code Documentation**:

  - Write clear, concise docstrings for all modules, classes, functions, and methods, following the Google Python Style Guide (see `docs/source/style_guide.rst`).
  - Include type hints in docstrings where applicable, complementing `mypy` annotations.
  - Example:

    ```python
    def calculate_profit(trades: List[Dict[str, float]]) -> float:
        """Calculate total profit from a list of trades.

        Args:
            trades: List of dictionaries containing trade details with 'amount' and 'price' keys.

        Returns:
            Total profit as a float.

        Raises:
            ValueError: If any trade dictionary is missing required keys.
        """
    ```

- **Project Documentation**:
  - Update Sphinx documentation in `docs/source/` for any new features, modules, or changes to existing functionality.
  - Build and verify documentation locally:
    ```bash
    cd docs
    make html
    ```
  - Ensure no warnings or errors in the build output (viewable in `docs/build/html/`).

### 5. Pre-commit Hook Compliance

- **Mandatory Checks**:

  - All contributions must pass pre-commit hooks, including `black`, `pylint`, `mypy`, `check-yaml`, `check-json`, and `check-eol`.
  - Install hooks during setup:
    ```bash
    pre-commit install
    ```
  - Run hooks manually to validate:
    ```bash
    pre-commit run --all-files
    ```
  - Review `pre-commit.log` for detailed error messages if hooks fail.

- **Custom Hooks**:
  - The project uses custom pre-commit hooks (defined in `.pre-commit-config.yaml`) for project-specific checks, such as validating configuration files or script syntax.
  - Ensure compatibility with the specified `pre-commit` version:
    ```bash
    pre-commit --version
    ```

### 6. Performance and Scalability

- **Optimize Code**:

  - Write efficient code, avoiding unnecessary complexity (e.g., excessive nested loops, redundant computations).
  - Profile performance-critical sections using tools like `cProfile` or `pytest-benchmark`:
    ```bash
    python -m cProfile -s time src/main.py
    ```
  - Address performance bottlenecks identified during testing or CI.

- **Scalability Considerations**:
  - Design features to handle reasonable input sizes and edge cases, as defined in the feature's use case.
  - Document any scalability limitations in docstrings or `docs/source/`.

### 7. Security Best Practices

- **Secure Coding**:

  - Avoid hardcoding sensitive information (e.g., API keys, credentials) in source code.
  - Use environment variables or configuration files (e.g., `config.yaml`) for sensitive data.
  - Validate all user inputs to prevent injection vulnerabilities (e.g., SQL injection, command injection).

- **Dependency Management**:
  - Pin dependency versions in `requirements.txt` to avoid breaking changes.
  - Check for known vulnerabilities using `pip-audit`:
    ```bash
    pip-audit
    ```
  - Update dependencies only after verifying compatibility with the project.

### 8. Additional Guidelines

- **Commit Messages**:

  - Write clear, concise commit messages following the Conventional Commits specification (e.g., `feat: add trade validation function`, `fix: resolve pylint F0401 error`).
  - Reference related issues (e.g., `Closes #123`) where applicable.

- **Code Review Expectations**:

  - Expect feedback on code style, functionality, and test coverage during pull request reviews.
  - Respond to review comments promptly and incorporate suggestions to align with project standards.

- **Troubleshooting**:
  - If pre-commit hooks or CI fail, check `pre-commit.log` or CI logs for details.
  - Consult the project's issue tracker or `README.md` for common issues and solutions.
  - Include relevant logs or error messages in pull request discussions or issue reports.

By adhering to these code quality standards, contributors help maintain the Categorical-Trading project's reliability, readability, and professionalism. For further details, refer to `README.md`, `.pylintrc`, `mypy.ini`, and `docs/source/style_guide.rst`.

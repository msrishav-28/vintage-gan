@echo off
REM VintageGAN Automation Script for Windows
REM Provides easy commands for development tasks

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="hooks" goto hooks
if "%1"=="test" goto test
if "%1"=="test-full" goto test-full
if "%1"=="verify" goto verify
if "%1"=="watch" goto watch
if "%1"=="quality" goto quality
if "%1"=="format" goto format
if "%1"=="lint" goto lint
if "%1"=="ci" goto ci
if "%1"=="all" goto all
if "%1"=="clean" goto clean
if "%1"=="results" goto results
if "%1"=="train" goto train
if "%1"=="train-auto" goto train-auto
if "%1"=="train-quick" goto train-quick
goto help

:help
echo VintageGAN Development Commands
echo ================================
echo.
echo Setup:
echo   automate install        Install package and dependencies
echo   automate hooks          Install git pre-commit hooks
echo.
echo Testing:
echo   automate test           Run quick tests
echo   automate test-full      Run all tests
echo   automate verify         Verify installation
echo   automate watch          Watch files and auto-test
echo.
echo Code Quality:
echo   automate quality        Run all quality checks
echo   automate format         Format code with black
echo   automate lint           Lint code with flake8
echo.
echo CI/CD:
echo   automate ci             Run full CI pipeline locally
echo   automate all            Run everything (format + quality + test)
echo.
echo Cleanup:
echo   automate clean          Remove generated files
echo.
echo Results:
echo   automate results        Generate all results for paper
echo.
echo Training (Automated):
echo   automate train          Interactive training setup
echo   automate train-auto     Fully automated training
echo   automate train-quick    Quick test (~30 min)
echo.
goto end

:install
echo Installing VintageGAN...
pip install -e .
pip install pre-commit black flake8 pytest
echo ✅ Installation complete!
goto end

:hooks
echo Installing git hooks...
python autotest.py --install-hooks
echo ✅ Git hooks installed!
goto end

:test
echo Running quick tests...
python run_tests.py --quick
goto end

:test-full
echo Running full test suite...
python run_tests.py
goto end

:verify
echo Verifying installation...
python verify_installation.py
goto end

:watch
echo Starting watch mode (Ctrl+C to stop)...
python autotest.py
goto end

:quality
echo Running code quality checks...
python autotest.py --once --quality
goto end

:format
echo Formatting code with black...
black . --line-length 100 --exclude="venv|build|dist|\.eggs"
echo ✅ Code formatted!
goto end

:lint
echo Linting code with flake8...
flake8 . --max-line-length=100 --exclude=venv,build,dist,.eggs --ignore=E203,W503
goto end

:ci
echo Running full CI pipeline locally...
python autotest.py --all --full
echo.
echo Testing package build...
python -m build
echo ✅ CI pipeline complete!
goto end

:all
echo Running format + quality + tests...
black . --line-length 100 --exclude="venv|build|dist|\.eggs"
python autotest.py --all --full
echo.
echo ✅ All checks passed!
goto end

:clean
echo Cleaning generated files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r %%d in (.pytest_cache) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r %%d in (.mypy_cache) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo ✅ Cleanup complete!
goto end

:results
echo Generating results for paper...
python compile_notebooks.py --format html pdf
echo ✅ Results generated in results/ directory!
goto end

:train
echo Starting interactive training setup...
python setup_and_train.py
goto end

:train-auto
echo Starting fully automated training...
python run_full_pipeline.py
goto end

:train-quick
echo Starting quick test training...
python run_full_pipeline.py --quick-test
goto end

:end

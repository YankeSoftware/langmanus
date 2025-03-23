@echo off
REM LangManus Launcher Script
TITLE LangManus AI System

echo:
echo ===================================
echo LangManus AI System Launcher
echo ===================================
echo:

:menu
echo Please select an option:
echo:
echo 1. Launch LangManus
echo 2. Run diagnostics
echo 3. Run LM Studio compatibility tests
echo 4. View recent feedback
echo 5. Exit
echo:

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto launch
if "%choice%"=="2" goto diagnostics
if "%choice%"=="3" goto tests
if "%choice%"=="4" goto feedback
if "%choice%"=="5" goto end

echo Invalid option, please try again.
goto menu

:launch
cls
echo Launching LangManus...
echo:
python main.py
echo:
echo Press any key to return to the menu...
pause >nul
cls
goto menu

:diagnostics
cls
echo Running system diagnostics...
echo:
python main.py --diagnostics
echo:
echo Press any key to return to the menu...
pause >nul
cls
goto menu

:tests
cls
echo Running LM Studio compatibility tests...
echo:
python test_lm_studio.py
echo:
echo Press any key to return to the menu...
pause >nul
cls
goto menu

:feedback
cls
echo Viewing recent feedback...
echo:
python -c "from src.integration import feedback; recent = feedback.get_recent_feedback(5); print(f'Average score: {feedback.get_average_score():.1f}/5.0\n') if feedback.get_average_score() else print('No feedback data available yet.\n'); [print(f'ID: {f[\"id\"]}\nScore: {f[\"score\"]}/5\nQuery: {f[\"user_input\"]}\nResponse: {f[\"model_response\"][:100]}...\nFeedback: {f[\"feedback_text\"]}\n---\n') for f in recent]"
echo:
echo Press any key to return to the menu...
pause >nul
cls
goto menu

:end
echo:
echo Thank you for using LangManus!
echo:
exit 
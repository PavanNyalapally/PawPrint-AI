# VS Code Pylance Import Resolution - Setup Guide

## Problem
Pylance shows "Import could not be resolved" warnings for:
- tensorflow
- keras
- numpy
- flask
- etc.

## Cause
VS Code/Pylance isn't configured to look in your virtual environment's `site-packages`.

## Solution Applied

### 1. Created `.vscode/settings.json`
Tells VS Code to use the venv Python interpreter and look in venv's site-packages for imports.

Key settings:
- `python.defaultInterpreterPath` - Points to venv Python
- `python.analysis.extraPaths` - Includes venv packages directory
- `python.analysis.exclude` - Ignores venv during analysis (prevents duplicate checks)

### 2. Created `pyrightconfig.json`
Pyright configuration for advanced type checking with correct Python path.

### 3. Created `.env`
Environment variables for development (optional but helpful).

## How to Apply

### Option A: Manual Setup (Recommended)
1. Close VS Code completely
2. Reopen the project folder
3. When prompted "Install Python packages", click "Yes"
4. Select the Python interpreter: 
   - Command Palette: `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
   - Type: "Python: Select Interpreter"
   - Choose: `./venv/bin/python`

### Option B: Quick Fix
1. Command Palette: `Ctrl+Shift+P` or `Cmd+Shift+P`
2. Type: "Python: Clear Python Diagnostics Cache"
3. Press Enter
4. Reload VS Code: `Ctrl+R` or `Cmd+R`

## Verification

All warnings should disappear:
- Train_model.py should show no import errors
- app.py should show no import errors
- Script files should show no import errors

Check the "Problems" panel:
- Should be empty or only have "reportMissingImports" warnings resolved

## If Issues Persist

### Check 1: Verify Interpreter
```bash
# Terminal in VS Code
which python
# or on Windows: where python

# Should show: /Users/nyalapallypavankumar/PawPrintAI/venv/bin/python
```

### Check 2: Verify Packages Installed
```bash
# Terminal in VS Code
pip list | grep -i tensorflow
# Should show: tensorflow version

pip list | grep -i flask
# Should show: Flask version
```

### Check 3: Reload Pylance
1. Command Palette: `Ctrl+Shift+P`
2. Type: "Pylance: Restart Pylance"
3. Click

### Check 4: Hard Reset
1. Delete `.vscode/` folder (will be recreated)
2. Close VS Code
3. Reopen project
4. Select interpreter again

## Expected Results

✅ No "Import could not be resolved" warnings
✅ Autocomplete works for imported modules
✅ Code navigation works (Go to Definition)
✅ Type hints work properly
✅ All diagnostics pass

## Files Created

- `.vscode/settings.json` - VS Code workspace settings
- `pyrightconfig.json` - Pyright type checker config  
- `.env` - Environment variables (optional)

These are already in `.gitignore`, so they won't be committed to version control.

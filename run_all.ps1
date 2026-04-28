Write-Host "Starting MLflow server in a new window..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\venv\Scripts\mlflow.exe server --host 0.0.0.0 --port 5000`""

Write-Host "Waiting 5 seconds for MLflow to start..."
Start-Sleep -Seconds 5

Write-Host "Starting Pipeline (Train -> Drift -> Retrain) in a new window..."
Start-Process powershell -ArgumentList "-NoExit -Command `"Write-Host '--- Running train.py ---'; .\venv\Scripts\python.exe train.py; Write-Host '--- Running drift_detector.py ---'; .\venv\Scripts\python.exe drift_detector.py; Write-Host '--- Running retrain.py ---'; .\venv\Scripts\python.exe retrain.py --auto`""

Write-Host "Starting FastAPI server in a new window..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000 --reload`""

Write-Host "All processes have been launched in separate windows."

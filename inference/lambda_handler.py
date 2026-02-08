import subprocess
import sys

def handler(event, context):
    """Lambda entry point."""
    print("Starting forecast pipeline...")
    
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        cwd="/app"
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with code {result.returncode}")
    
    return {
        'statusCode': 200,
        'body': 'Forecast complete'
    }
import subprocess
result = subprocess.run(["pip", "install", "--upgrade", "scipy"], capture_output=True, text=True)

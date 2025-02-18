import subprocess

print("running: prep_data.py...")
subprocess.run(["python", "prep_data.py"], check=True)

print("running: train.py...")
subprocess.run(["python", "train.py"], check=True)

print("all process completed!")

import os
import subprocess
from datetime import datetime
from fastapi import FastAPI, HTTPException
import json
import os

app = FastAPI()

@app.post("/run")
def run_task(task: str):
    task_lower = task.lower()
    if "install uv" in task_lower:
        return install_uv()
    elif "format" in task_lower and "prettier" in task_lower:
        return format_file_with_prettier()
    elif "count wednesdays" in task_lower:
        return count_wednesdays()
    elif "sort contacts" in task.lower():
        return sort_contacts()
    else:
        raise HTTPException(status_code=400, detail="Task not recognized")

def install_uv():
    try:
        subprocess.run(["pip", "install", "uv"], check=True)
        return {"message": "uv installed successfully"}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to install uv")

def format_file_with_prettier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "format.md"))

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {file_path} not found")

    try:
        subprocess.run(["C:\\Program Files\\nodejs\\npx.cmd", "prettier@3.4.2", "--write", file_path], check=True)
        return {"message": f"Formatted {file_path} using Prettier"}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Prettier formatting failed")

def count_wednesdays():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "dates.txt"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "data", "dates-wednesdays.txt"))

    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_path} not found")

        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]  # Strip empty lines

        wednesday_count = sum(1 for date in lines if datetime.strptime(date, "%Y-%m-%d").weekday() == 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
        with open(output_path, "w") as f:
            f.write(str(wednesday_count))

        return {"message": f"Counted {wednesday_count} Wednesdays in {file_path}"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format in {file_path}: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def sort_contacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "../data/contacts.json")
    output_path = os.path.join(base_dir, "../data/contacts-sorted.json")

    try:
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"File {input_path} not found")

        # Read the JSON file
        with open(input_path, "r", encoding="utf-8") as f:
            contacts = json.load(f)

        # Sort by last_name, then first_name
        sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

        # Write sorted contacts back to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_contacts, f, indent=4)

        return {"message": f"Sorted contacts and saved to {output_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing contacts: {str(e)}")

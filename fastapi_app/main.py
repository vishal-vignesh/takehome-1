import os
import subprocess
from datetime import datetime
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import json
import os
import glob

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
    elif "logs recent" in task.lower():  # For task A5
        return write_recent_log_lines()
    elif "markdown" in task.lower() and "index" in task.lower():
        return extract_h1_from_markdown()
    elif "extract" in task_lower and "email" in task_lower:
        return extract_email()
    elif "card" in task_lower or "credit card" in task_lower:
        return extract_credit_card()
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

def write_recent_log_lines():
    logs_dir = "../data/logs/"
    output_file = "../data/logs-recent.txt"
    
    try:
        # Get all .log files in the logs directory
        log_files = glob.glob(os.path.join(logs_dir, "*.log"))
        
        # Sort the files by modification time, most recent first
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Take the 10 most recent log files
        recent_log_files = log_files[:10]
        
        # Collect the first lines from these files
        first_lines = []
        for log_file in recent_log_files:
            with open(log_file, "r") as f:
                first_line = f.readline().strip()  # Get first line
                first_lines.append(first_line)
        
        # Write the first lines to the output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists
        with open(output_file, "w") as f:
            f.write("\n".join(first_lines))  # Join lines with a newline character
        
        return {"message": f"Successfully wrote first lines of recent log files to {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing log files: {str(e)}")

def extract_h1_from_markdown():
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/docs/")
    index_file = os.path.join(docs_dir, "index.json")
    index_data = {}

    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"Directory {docs_dir} not found")

    for file_name in os.listdir(docs_dir):
        if file_name.endswith(".md"):
            file_path = os.path.join(docs_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("# "):  # First H1 header found
                        index_data[file_name] = line[2:]  # Remove '# ' from title
                        break  # Stop reading once the first H1 is found

    # Write the index to index.json
    with open(index_file, "w", encoding="utf-8") as json_file:
        json.dump(index_data, json_file, indent=4)

    return {"message": f"Index file created at {index_file}"}

def extract_email():
    input_file = "../data/email.txt"
    output_file = "../data/email-sender.txt"
    
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Email file not found")
    
    with open(input_file, "r") as f:
        email_content = f.read().strip()

    # Call GPT-4o-Mini using AI Proxy (replace with actual API call)
    sender_email = call_llm_to_extract_email(email_content)

    if not sender_email:
        raise HTTPException(status_code=400, detail="Failed to extract email address")

    with open(output_file, "w") as f:
        f.write(sender_email)

    return {"message": f"Extracted sender email and saved to {output_file}"}

def call_llm_to_extract_email(email_content):
    """Mock function to simulate an LLM call for email extraction."""
    # In real implementation, use GPT-4o-Mini via AI Proxy
    import re
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_content)
    return match.group(0) if match else None

def extract_credit_card():
    image_path = "../data/credit-card.png"
    output_path = "../data/credit-card.txt"

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"File {image_path} not found")

    try:
        # Initialize OpenAI client (Use AIPROXY_TOKEN)
        client = OpenAI(api_key=os.environ["AIPROXY_TOKEN"])

        # Read image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Send image to OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract the credit card number from this image and return only the digits without spaces."},
                {"role": "user", "content": image_data}
            ]
        )

        # Get extracted text
        extracted_text = response.choices[0].message.content.strip()
        card_number = extracted_text.replace(" ", "").replace("-", "")  # Remove spaces and dashes

        # Save to file
        with open(output_path, "w") as f:
            f.write(card_number)

        return {"message": f"Extracted credit card number and saved to {output_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting credit card: {str(e)}")
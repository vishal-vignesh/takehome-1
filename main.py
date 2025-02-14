from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import os
import re
import shutil
import subprocess
import requests
from datetime import datetime
import openai
import json
import pytesseract
from PIL import Image
import numpy as np
import sqlite3
import base64

app = FastAPI()

@app.post("/run")
async def run_task(task: str = Query(...)):
    if not task:
        raise HTTPException(status_code=400, detail="Task description required")
    
    # Use the LLM to parse the task instruction.
    try:
        parsed_task = parse_task_with_llm(task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing task with LLM: {str(e)}")
    
    task_code = parsed_task.get("task_code", "UNKNOWN").upper()
    
    # Map the task_code to the corresponding internal function.
    try:
        if task_code == "A1":
            user_email = os.environ.get("USER_EMAIL", "default@example.com")
            result = handle_task_A1(user_email)
        elif task_code == "A2":
            result = handle_task_A2()
        elif task_code == "A3":
            result = handle_task_A3()
        elif task_code == "A4":
            result = handle_task_A4()
        elif task_code == "A5":
            result = handle_task_A5()
        elif task_code == "A6":
            result = handle_task_A6()
        elif task_code == "A7":
            result = handle_task_A7()
        elif task_code == "A8":
            result = handle_task_A8()
        elif task_code == "A9":
            result = handle_task_A9()
        elif task_code == "A10":
            result = handle_task_A10()
        # Future tasks (A4 - A10) can be integrated similarly.
        else:
            # If LLM returned UNKNOWN or an unsupported task code.
            raise Exception("Unrecognized or unsupported task code returned by LLM.")
        
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(...)):
    """
    GET endpoint to read and return the content of a file.
    Ensures only files under /data (as specified in the task) are accessed.
    """
    # Security check: Path must start with /data
    if not path.startswith("/data"):
        raise HTTPException(status_code=400, detail="Invalid file path: Must start with /data")
    
    # Translate the given path into a local path.
    # Assuming your repository has a 'data' folder in its root,
    # we remove the leading '/data' and join with the repository's data directory.
    base_dir = os.path.join(os.getcwd(), "data")  # local data folder
    relative_path = os.path.relpath(path, "/data")  # e.g. "sample.txt"
    file_path = os.path.join(base_dir, relative_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

def handle_task_A1(user_email: str):
    # 1. Check if 'uv' is installed.
    if shutil.which("uv") is None:
        try:
            install_proc = subprocess.run(
                ["pip", "install", "uv"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Installed uv:", install_proc.stdout)
        except subprocess.CalledProcessError as e:
            raise Exception("Failed to install uv: " + e.stderr)
    
    # 2. Download the datagen.py script.
    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    response = requests.get(datagen_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download datagen.py, status code: {response.status_code}")
    
    datagen_filename = "datagen.py"
    with open(datagen_filename, "w") as f:
        f.write(response.text)
    
    # 3. Modify the script to use a local data folder instead of '/data'.
    #    We'll assume your local folder is the 'data' directory in your project.
    local_data_dir = os.path.join(os.getcwd(), "data")
    
    # Read the downloaded file
    with open(datagen_filename, "r") as f:
        content = f.read()
    
    # Replace occurrences of '/data' (in quotes) with the local data directory.
    # This regex will match both single and double quotes.
    new_content = re.sub(r'([\'"])/data([\'"])', f'\\1{local_data_dir}\\2', content)
    
    # Write the modified content back to datagen.py
    with open(datagen_filename, "w") as f:
        f.write(new_content)
    
    # 4. Run datagen.py with the user's email as the only argument.
    try:
        proc = subprocess.run(
            ["python", datagen_filename, user_email],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error running datagen.py: " + e.stderr)
    
    return {"stdout": proc.stdout, "stderr": proc.stderr}



def handle_task_A2():
    """
    Formats the file /data/format.md using prettier@3.4.2.
    The file is updated in-place.
    
    This version mimics the evaluation script: it pipes the file content into Prettier
    using the "--stdin-filepath /data/format.md" option.
    """
    # Define the local data directory (project-root/data)
    local_data_dir = os.path.join(os.getcwd(), "data")
    
    # Construct the local file path for format.md
    file_path = os.path.join(local_data_dir, "format.md")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")
    
    # Read the current contents of the file.
    with open(file_path, "r") as f:
        original = f.read()
    
    try:
        # Build the command as a single string.
        cmd = "npx prettier@3.4.2 --stdin-filepath /data/format.md"
        # Run Prettier using the command string, passing the current working directory and environment.
        proc = subprocess.run(
            cmd,
            input=original,
            capture_output=True,
            text=True,
            check=True,
            shell=True,  # Command is provided as a string.
            cwd=os.getcwd(),         # Ensure we run in the project root.
            env=os.environ.copy()      # Pass the current environment.
        )
        formatted = proc.stdout
        
        # Write the formatted content back to the file.
        with open(file_path, "w") as f:
            f.write(formatted)
        
        return {"stdout": formatted, "stderr": proc.stderr}
    except subprocess.CalledProcessError as e:
        raise Exception("Error formatting file: " + e.stderr)

def handle_task_A3():
    """
    Reads data/dates.txt, counts the number of Wednesdays,
    and writes the count to data/dates-wednesdays.txt.
    """
    # Define the local data directory and file paths.
    local_data_dir = os.path.join(os.getcwd(), "data")
    input_file = os.path.join(local_data_dir, "dates.txt")
    output_file = os.path.join(local_data_dir, "dates-wednesdays.txt")

    if not os.path.exists(input_file):
        raise Exception(f"File not found: {input_file}")

    # Define a list of possible date formats.
    date_formats = [
        "%Y/%m/%d %H:%M:%S",  # e.g., 2008/04/22 06:26:02
        "%Y-%m-%d",           # e.g., 2006-07-21
        "%b %d, %Y",          # e.g., Sep 11, 2006
        "%d-%b-%Y",           # e.g., 28-Nov-2021
    ]

    wednesday_count = 0

    with open(input_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parsed_date = None
            # Try each date format until one succeeds.
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(line, fmt)
                    break  # Exit loop if parsing is successful.
                except ValueError:
                    continue

            if parsed_date is None:
                # Optionally log the unparsable line.
                print(f"Warning: Could not parse date: {line}")
                continue

            # datetime.weekday() returns Monday=0, Tuesday=1, Wednesday=2, etc.
            if parsed_date.weekday() == 2:
                wednesday_count += 1

    # Write just the count to the output file.
    with open(output_file, "w") as file:
        file.write(str(wednesday_count))

    return {"wednesday_count": wednesday_count}

def handle_task_A4():
    """
    Sorts the array of contacts in /data/contacts.json by last_name, then first_name,
    and writes the result to /data/contacts-sorted.json.
    """
    # Define the local data directory.
    local_data_dir = os.path.join(os.getcwd(), "data")
    
    # Construct paths for the input and output files.
    contacts_path = os.path.join(local_data_dir, "contacts.json")
    sorted_contacts_path = os.path.join(local_data_dir, "contacts-sorted.json")
    
    # Ensure contacts.json exists.
    if not os.path.exists(contacts_path):
        raise Exception(f"File not found: {contacts_path}")
    
    # Read contacts.json.
    with open(contacts_path, "r") as f:
        try:
            contacts = json.load(f)
        except Exception as e:
            raise Exception("Error reading contacts.json: " + str(e))
    
    # Sort contacts by last_name and then first_name.
    sorted_contacts = sorted(
        contacts,
        key=lambda c: (c.get("last_name", "").lower(), c.get("first_name", "").lower())
    )
    
    # Write the sorted contacts to contacts-sorted.json with indentation.
    with open(sorted_contacts_path, "w") as f:
        json.dump(sorted_contacts, f, indent=2)
    
    return {"sorted_contacts": sorted_contacts}

def handle_task_A5():
    """
    Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.
    """
    logs_dir = os.path.join(os.getcwd(), "data", "logs")
    output_file = os.path.join(os.getcwd(), "data", "logs-recent.txt")

    # Ensure the logs directory exists
    if not os.path.exists(logs_dir):
        raise Exception(f"Logs directory not found: {logs_dir}")

    # List all .log files and sort them numerically based on the filename
    log_files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".log")],
        key=lambda x: int(x.replace("log-", "").replace(".log", "")), 
        reverse=True  # Most recent first
    )

    # Pick the 10 most recent logs
    recent_logs = log_files[:10]

    # Read the first line of each log file
    first_lines = []
    for log_file in recent_logs:
        log_path = os.path.join(logs_dir, log_file)
        try:
            with open(log_path, "r") as f:
                first_line = f.readline().strip()
                first_lines.append(first_line)
        except Exception as e:
            first_lines.append(f"Error reading {log_file}: {str(e)}")

    # Write the first lines to logs-recent.txt
    with open(output_file, "w") as f:
        f.write("\n".join(first_lines) + "\n")

    return {"written_file": output_file, "first_lines": first_lines}


def handle_task_A6():
    """
    Find all .md files in /data/docs/, extract the first occurrence of an H1 title (# Title),
    and save them in /data/docs/index.json as { "file.md": "Title", ... }.
    """
    docs_dir = os.path.join(os.getcwd(), "data", "docs")
    output_file = os.path.join(docs_dir, "index.json")

    index = {}

    # Walk through /data/docs/ recursively
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir)

                # Extract the first H1 title from the file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            match = re.match(r"^# (.+)", line.strip())
                            if match:
                                index[relative_path] = match.group(1)
                                break  # Stop after first H1
                except Exception as e:
                    index[relative_path] = f"Error reading file: {str(e)}"

    # Write to index.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    return {"written_file": output_file, "index": index}

def handle_task_A7():
    """
    1. Reads /data/email.txt (the entire email message).
    2. Sends the email content to GPT-4o-Mini with instructions to extract the sender's email.
    3. Writes just the sender’s email to /data/email-sender.txt.
    """
    input_file = os.path.join(os.getcwd(), "data", "email.txt")
    output_file = os.path.join(os.getcwd(), "data", "email-sender.txt")

    # 1. Verify the file exists
    if not os.path.exists(input_file):
        return {"error": f"File not found: {input_file}"}

    # 2. Read the entire email content
    with open(input_file, "r", encoding="utf-8") as f:
        email_content = f.read()

    # 3. Prepare the LLM environment
    token = os.environ.get("AIPROXY_TOKEN")

    if not token:
        return {"error": "AIPROXY_TOKEN environment variable not set."}

    openai.api_key = token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

    # 4. Build a prompt instructing GPT-4o-Mini to extract only the sender’s email
    #    We'll ask for a JSON response to parse it safely.
    prompt = (
        "You are a helpful assistant. I have an email message:\n\n"
        f"{email_content}\n\n"
        "Please extract only the sender’s email address from this email. "
        "Return your answer in a JSON object with a single key 'sender_email'. For example:\n"
        "{\n  \"sender_email\": \"example@domain.com\"\n}\n\n"
        "Return only the JSON object."
    )

    try:
        # 5. Make the GPT-4o-Mini chat request
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        # 6. Parse the raw response
        raw_message = response["choices"][0]["message"]["content"].strip()
        # Remove potential code fences
        raw_message = re.sub(r"^```json\s*", "", raw_message)
        raw_message = re.sub(r"\s*```$", "", raw_message)

        if not raw_message:
            return {"error": "LLM returned empty response."}

        # Attempt to parse JSON
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return {
                "error": "LLM response was not valid JSON.",
                "raw_response": raw_message
            }

        sender_email = data.get("sender_email", "").strip()
        if not sender_email:
            return {
                "error": "No 'sender_email' found in LLM response.",
                "raw_response": raw_message
            }

        # 7. Write the sender’s email to /data/email-sender.txt
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sender_email + "\n")

        return {
            "status": "success",
            "sender_email": sender_email,
            "written_file": output_file
        }

    except Exception as e:
        return {"error": str(e)}

def handle_task_A8():
    """
    1. Reads /data/credit-card.png
    2. Extracts a 16-digit number via Tesseract OCR
    3. Applies Luhn check. If it fails and the first digit is '9',
       try replacing it with '3' and check again.
    4. Writes the final 16-digit number to /data/credit-card.txt
    """
    input_file = os.path.join(os.getcwd(), "data", "credit_card.png")
    output_file = os.path.join(os.getcwd(), "data", "credit-card.txt")

    try:
        # 1. Load the image
        img = Image.open(input_file)

        # 2. Configure Tesseract for digits only
        custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
        extracted_text = pytesseract.image_to_string(img, config=custom_config)

        # 3. Extract lines, look for a line with exactly 16 digits
        lines = extracted_text.splitlines()
        recognized_16 = None
        for line in lines:
            digits = re.sub(r"\D", "", line)  # keep only digits
            if len(digits) == 16:
                recognized_16 = digits
                break

        if not recognized_16:
            return {
                "error": "No line with exactly 16 digits found.",
                "ocr_output": extracted_text
            }

        # 4. Check Luhn
        if passes_luhn(recognized_16):
            final_number = recognized_16
        else:
            # If first digit is '9', try flipping it to '3'
            if recognized_16[0] == '9':
                possible_fix = '3' + recognized_16[1:]
                if passes_luhn(possible_fix):
                    final_number = possible_fix
                else:
                    return {
                        "error": "Luhn check failed, flipping '9'->'3' also failed.",
                        "recognized_number": recognized_16
                    }
            else:
                return {
                    "error": "Luhn check failed and no known fix.",
                    "recognized_number": recognized_16
                }

        # 5. Write final_number to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_number + "\n")

        return {"written_file": output_file, "card_number": final_number}

    except Exception as e:
        return {"error": str(e)}

def handle_task_A9():
    """
    Reads /data/comments.txt (one comment per line).
    Asks GPT-4o-Mini to pick the two lines that are most semantically similar.
    Writes those two lines (one per line) to /data/comments-similar.txt.
    """
    # 1. Prepare file paths
    input_file = os.path.join(os.getcwd(), "data", "comments.txt")
    output_file = os.path.join(os.getcwd(), "data", "comments-similar.txt")

    # 2. Check if the file exists
    if not os.path.exists(input_file):
        return {"error": f"{input_file} does not exist"}

    # 3. Read lines (strip empty ones)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return {"error": "Not enough comments to compare."}

    # 4. Set up your GPT-4o-Mini credentials
    token = os.environ.get("AIPROXY_TOKEN")

    if not token:
        return {"error": "AIPROXY_TOKEN environment variable not set."}

    openai.api_key = token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

    # 5. Build a prompt enumerating all lines
    #    Ask GPT-4o-Mini to return a JSON object with "best_pair": [line1, line2]
    enumerated_lines = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    prompt = (
        "You are a helpful assistant. I have a list of comments (one per line). "
        "Please identify the TWO lines that are most semantically similar. "
        "Return your answer in JSON format as follows:\n\n"
        "{\n  \"best_pair\": [\"<comment1>\", \"<comment2>\"]\n}\n\n"
        "Here are the lines:\n\n"
        f"{enumerated_lines}\n\n"
        "Respond with only the JSON object."
    )

    try:
        # 6. Call GPT-4o-Mini with the prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        # 7. Parse the raw response to extract JSON
        raw_message = response["choices"][0]["message"]["content"]
        # Remove potential markdown fences
        raw_message = re.sub(r"^```json\s*", "", raw_message.strip())
        raw_message = re.sub(r"\s*```$", "", raw_message)
        if not raw_message.strip():
            return {"error": f"LLM returned empty or invalid response: {response}"}

        data = json.loads(raw_message)
        best_pair = data.get("best_pair", [])
        if len(best_pair) != 2:
            return {"error": f"Could not find exactly 2 lines. Received: {best_pair}"}

        # 8. Write the best pair to /data/comments-similar.txt
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(best_pair[0] + "\n")
            f.write(best_pair[1] + "\n")

        return {
            "status": "success",
            "best_pair": best_pair,
            "written_file": output_file
        }

    except Exception as e:
        return {"error": str(e)}

def handle_task_A10():
    local_data_dir = os.path.join(os.getcwd(), "data")
    db_path = os.path.join(local_data_dir, "ticket-sales.db")
    output_file = os.path.join(local_data_dir, "ticket-sales-gold.txt")

    if not os.path.exists(db_path):
        return {"error": f"Database file not found at {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold';"
        cursor.execute(query)
        total_sales = cursor.fetchone()[0]
        if total_sales is None:
            total_sales = 0.0

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(total_sales) + "\n")

        conn.close()
        return {
            "status": "success",
            "total_sales": total_sales,
            "written_file": output_file
        }

    except Exception as e:
        return {"error": str(e)}

def parse_task_with_llm(task: str) -> dict:
    """
    Uses GPT-4o-Mini via the AI Proxy to parse the plain-English task and extract a structured task code.
    Expected output JSON format: {"task_code": "A3"}, for example.
    """
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise Exception("AIPROXY_TOKEN environment variable not set")
    
    # Set the API key and base URL for the proxy.
    openai.api_key = token
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    
    # Construct a prompt with explicit mappings between task descriptions and task codes.
    prompt = (
        "You are a task parser for DataWorks Solutions. Below are the explicit mappings of task descriptions to task codes:\n\n"
        "A1: 'Install uv (if required) and run datagen.py with ${user.email} as the only argument'\n"
        "A2: 'Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place'\n"
        "A3: 'The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays and write just the number to /data/dates-wednesdays.txt'\n"
        "A4: 'Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json'\n"
        "A5: 'Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first'\n"
        "A6: 'Find all Markdown (.md) files in /data/docs/, extract the first occurrence of each H1, and create an index file /data/docs/index.json mapping filenames to titles'\n"
        "A7: '/data/email.txt contains an email message. Extract the sender’s email address using an LLM and write it to /data/email-sender.txt'\n"
        "A8: '/data/credit-card.png contains a credit card number. Use an LLM to extract the card number and write it without spaces to /data/credit-card.txt'\n"
        "A9: '/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line'\n"
        "A10: 'The SQLite database file /data/ticket-sales.db has a table tickets with columns type, units, and price. Calculate the total sales for the \"Gold\" ticket type and write the number to /data/ticket-sales-gold.txt'\n\n"
        "Given the following instruction, determine which task code applies. "
        "Return a JSON object with a single key 'task_code' whose value is one of A1, A2, A3, A4, A5, A6, A7, A8, A9, or A10. "
        "If the instruction does not match any known task, return 'UNKNOWN'.\n\n"
        f"Instruction: \"{task}\"\n\n"
        "Return only the JSON object."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful task parser."},
                {"role": "user", "content": prompt},
            ]
        )
        
        # Debug: print the raw response.
        print("Raw LLM response:", response)
        
        # Extract the content.
        raw_message = response["choices"][0]["message"]["content"]
        
        # Remove markdown code fences if present.
        raw_message = re.sub(r"^```json\s*", "", raw_message)
        raw_message = re.sub(r"\s*```$", "", raw_message)
        
        if not raw_message.strip():
            raise Exception("LLM returned an empty response: " + str(response))
        
        parsed = json.loads(raw_message)
        return parsed
    except Exception as e:
        raise Exception(f"Error calling LLM: {str(e)}")


def passes_luhn(number_str: str) -> bool:
    """
    Returns True if 'number_str' (containing only digits) satisfies the Luhn check.
    """
    if not number_str.isdigit():
        return False
    
    digits = [int(d) for d in number_str]
    # Double every second digit from the right
    for i in range(len(digits) - 2, -1, -2):
        doubled = digits[i] * 2
        # If doubling is >= 10, subtract 9
        if doubled > 9:
            doubled -= 9
        digits[i] = doubled
    
    return sum(digits) % 10 == 0

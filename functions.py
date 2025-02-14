### count_days
### extract_pkgname
## extract dayname
import re
import requests
from fuzzywuzzy import fuzz
from datetime import datetime
from openai import OpenAI
def get_task_output(AIPROXY_TOKEN, task):
    client = OpenAI(api_key = AIPROXY_TOKEN)
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": task}
    ]
    )
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()
def count_days(dayname:str):
    ## count sundays instead of sunday
    days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    dayvalue = -1
    day = None
    for d in days:
        if d in dayname.lower():
            dayvalue = days[d]
            day = d
            break
    try:
        print("This will not cause any error")
        with open("/data/dates.txt","r") as file:
            print("My line")
            data = file.readlines()
            count  = sum([1 for line in data if datetime.strptime( line.strip(),"%Y-%m-%d").weekday()==dayvalue])
            f = open("/data/{}-count".format(day), "w")
            f.write(str(count))
            f.close()
    except:
        print("There is no File in data directory try making one")
def extract_dayname(task:str):
    match = re.search(r'count\s+(\w+)',task)
    if match:
        return match.group(1)
    return ""
def extract_package(task:str):
    match = re.search(r'install\s+(\w+)',task)
    if match:
        return match.group(1)
    return ""
def get_correct_pkgname(pkgname: str):
    with open("packages.txt","r",encoding="utf-8") as file:
        data = file.read().strip()
        packages = [line.strip() for line in data.split(" ")]
        corrects = []
        for pkg in packages:
            if fuzz.ratio(pkgname, pkg) >= 90:
                corrects.append(pkg)
        if corrects:
            if len(corrects)==1:
                return corrects[0]
            elif len(corrects)>=2:
                return corrects[-1]
        return ""
import requests
from bs4 import BeautifulSoup
url = 'https://pypi.org/simple/'
def getpackages(url):
    response = requests.get(url)
    if response.status_code==200:
        soup = BeautifulSoup(response.text, "html.parser")
        packages = [a.text.strip() for a in soup.find_all('a')]
        with open("packages.txt","w+") as file:
            for pkg in packages:
                file.write(str(pkg)+" ")
            file.close()
    else:
        print("We can't fetch it right now!")
getpackages(url)
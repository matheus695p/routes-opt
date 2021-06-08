import requests
from bs4 import BeautifulSoup


def ask_google(city1, city2, question="distancia entre"):
    # url donde ir a buscar la info de google
    url = "https://www.google.com/search?q="+question+f"{city1}"+f"+{city2}"
    print(url)
    # hacer request de html
    HTML = requests.get(url)
    # parsear html
    soup = BeautifulSoup(HTML.text, 'html.parser')
    # encontrar el div que tenga el precio
    text = soup.find("div", attrs={'class': 'BNeawe iBp4i AP7Wnd'}).find(
        "div", attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
    print(text)
    return text

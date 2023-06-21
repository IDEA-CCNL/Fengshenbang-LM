import json
import requests
from bs4 import BeautifulSoup


def get_model_downloads(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to get response from URL")

    soup = BeautifulSoup(response.content, 'html.parser')
    model_data_node = soup.find_all('div', attrs={"class": "SVELTE_HYDRATER"})[3]
    data = json.loads(model_data_node['data-props'])
    all_downloads = 0
    for item in data['repos']:
        if 'downloads' not in item:
            item['downloads'] = 0
        all_downloads += item['downloads']
    return all_downloads


def main():
    all_downloads = get_model_downloads('https://huggingface.co/IDEA-CCNL?sort_models=downloads#models')
    print(f'Total downloads: {all_downloads}')


if __name__ == '__main__':
    main()

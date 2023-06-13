import json
import requests
from bs4 import BeautifulSoup

response = requests.get('https://huggingface.co/IDEA-CCNL?sort_models=downloads#models')
soup = BeautifulSoup(response.content, 'html.parser')
model_data_node = soup.find_all('div', attrs={"class": "SVELTE_HYDRATER"})[3]
data = json.loads(model_data_node['data-props'])
all_downloads = 0
for item in data['repos']:
    if 'downloads' not in item:
        item['downloads'] = 0
    all_downloads += item['downloads']
    print('name: {}, author: {}, downloads: {}, likes: {}'.format(
        item['id'], item['author'], item['downloads'], item['likes']))
print('total downloads {}'.format(all_downloads))

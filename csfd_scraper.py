import requests
import time
import re
import lxml.html
import utils
from pathlib import Path
import csv

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
DATA_PATH = Path('data/csfd').resolve()
MAX_REVIEWS = 4000
MAX_RETRIES = 5


def get_html(session: requests.Session, url: str):
    for i in range(MAX_RETRIES):
        try:
            response = session.get(url)
            response.raise_for_status()
            return lxml.html.fromstring(response.text)
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}, retrying... ({i + 1}/{MAX_RETRIES})')
            time.sleep(5)

    raise Exception(f"Failed to get '{url}', max retries exceeded.")


def init_directories():
    DATA_PATH.mkdir(parents=True, exist_ok=True)


def get_username_from_url(url: str) -> str:
    return url.split('uzivatel/', maxsplit=1)[-1].split('/', maxsplit=1)[0]


def get_movie_id_from_url(url: str) -> str:
    pattern = r"film/(\d+)(?:-[^/]+)?(?:/(\d+))?"
    matches = re.search(pattern, url)
    if matches:
        movie_id = matches.group(1)
        episode_id = matches.group(2)
        return f"{movie_id}_{episode_id}" if episode_id else movie_id
    raise Exception(f"Failed to get movie ID from '{url}'.")


def parse_review_count(text: str) -> int:
    for space in (' ', '\xa0'):
        text = text.replace(space, '')
    return int(text.split('rec', maxsplit=1)[0])


def get_users(session: requests.Session) -> list[str]:
    response = session.get('https://www.csfd.cz/uzivatele/nejaktivnejsi/?country=1')
    html = lxml.html.fromstring(response.text)

    user_urls = html.xpath('//h2[text()="Recenze"]/../following-sibling::div//div[@class="article-content"]//a[@class="user-title-name"]//@href')
    return [get_username_from_url(url) for url in user_urls]


def get_reviews(session: requests.Session, username: str, writer: csv.writer):
    page = 1
    reviews = 0

    while reviews < MAX_REVIEWS:
        #print(f'Going to page #{page}')
        url = f'https://www.csfd.cz/uzivatel/{username}/recenze/?page={page}'
        html = get_html(session, url)

        for review in html.xpath('//section[@class="box striped-articles user-reviews"]/div[@class="box-content"]/article'):
            movie_url = review.xpath('.//a[@class="film-title-name"]/@href')[0]
            movie_id = get_movie_id_from_url(movie_url)
            review_text = ''.join(review.xpath('.//div[@class="user-reviews-text"]//span[@class="comment"]')[0].itertext())
            review_text = utils.parse_author_text(review_text, False)
            if review_text:
                writer.writerow([username, movie_id, review_text])
                reviews += 1
            #print(f'Parsed review {movie_id}')

        if not html.xpath('//div[@class="user-main-content"]//a[@class="page-next"]'):
            return

        page += 1
        time.sleep(0.2)


def main():
    init_directories()

    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    users = get_users(session)
    user_count = len(users)

    filename = DATA_PATH / f'csfd_{time.strftime("%y%m%d%H%M%S")}.csv'

    with filename.open('a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['author', 'movie_id', 'text'])  # Header
        for i, user in enumerate(users):
            get_reviews(session, user, writer)
            print(f'Finished user {user} ({i+1}/{user_count})')

    print('Done')


if __name__ == '__main__':
    main()

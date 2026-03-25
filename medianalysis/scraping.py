from newspaper import Article
import re, random, time, requests as rq
import nltk
from waybackpy import WaybackMachineCDXServerAPI
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt')

from .distrib import BaseWorker

class scraper:
    def __init__(self, url, timeout, user_agent = None):
        self.timeout = timeout
        self.headers = {'user-agent': user_agent or 'Mozilla/5.0'}
        self.html, self.flag = self.autohandler(url)

    def vibe_check(self, url):
        try:
            time.sleep(random.uniform(0.3,1.7))

            GET = rq.get(url, headers = self.headers, timeout = self.timeout)
            HEADERS = GET.headers['Content-Type']
            MIMETYPE = HEADERS.split(';')[0]
            CHARSET = HEADERS.split('charset=')[-1]
            STATUS = GET.status_code

            if STATUS == 200 and 'text/html' in MIMETYPE:
                return GET.content.decode(CHARSET), True
            else:
                return None, False
        except:
            return None, False
            
    def autohandler(self, url):
        html, flag = self.vibe_check(url)

        if flag:
            return html, flag

        cdx = iter(
            WaybackMachineCDXServerAPI(
                url,
                user_agent = self.headers['user-agent']
            ).snapshots()
        )

        while True:
            try:
                snapshot = next(cdx)
            except StopIteration:
                break
            except Exception as e:
                continue

            try:
                html, flag = self.vibe_check(snapshot.archive_url)
                if flag:
                    return html, flag
            except:
                continue
        return None, False
        
    def extract_info(self):
        if not self.flag:
            return None, None, 'Unhandleable.'
        else:
            article = Article('',language='es')
            article.download(input_html = self.html)
            article.parse()
            return re.sub(r'\n\n',' ',article.text_cleaned), article.meta_keywords, None

class ScraperWorker(BaseWorker):
    def __init__(self, *args, timeout = 25, user_agent = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout
        self.headers = {'user-agent': user_agent or 'Mozilla/5.0'}

    def process_row(self, row):
        body, kws, err = scraper(
            row['url'],
            timeout=self.timeout,
            user_agent=self.headers['user-agent']
        ).extract_info()

        return {
            "media_name": row["media_name"],
            "publish_date": row["publish_date"],
            "title": row["title"],
            "url": row['url'],
            "body": body,
            "keywords": kws,
            "error": err,
        }
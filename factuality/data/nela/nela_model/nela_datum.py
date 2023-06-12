class NelaDatum:
    def __init__(self):
        self._id = None
        self._date = None
        self._source = None
        self._title = None
        self._content = None
        self._author = None
        self._url = None
        self._published = None
        self._published_utc = None
        self._collection_utc = None

    def id(self):
        return self._id
    
    def date(self):
        return self._date
    
    def source(self):
        return self._source
    
    def title(self):
        return self._title
    
    def content(self):
        return self._content
    
    def author(self):
        return self._author
    
    def url(self):
        return self._url
    
    def published(self):
        return self._published
    
    def published_utc(self):
        return self._published_utc
    
    def collection_utc(self):
        return self._collection_utc
    
    def set_id(self, id):
        self._id = id

    def set_date(self, date):
        self._date = date

    def set_source(self, source):
        self._source = source

    def set_title(self, title):
        self._title = title

    def set_content(self, content):
        self._content = content

    def set_author(self, author):
        self._author = author

    def set_url(self, url):
        self._url = url

    def set_published(self, published):
        self._published = published

    def set_published_utc(self, published_utc):
        self._published_utc = published_utc

    def set_collection_utc(self, collection_utc):
        self._collection_utc = collection_utc

    def to_dict(self):
        return {
            'id': self.id(),
            'date': self.date(),
            'source': self.source(),
            'title': self.title(),
            'content': self.content(),
            'author': self.author(),
            'url': self.url(),
            'published': self.published(),
            'published_utc': self.published_utc(),
            'collection_utc': self.collection_utc(),
        }
    
    @staticmethod
    def from_dict(val):
        datum = NelaDatum()
        datum.set_id(val['id'])
        datum.set_date(val['date'])
        datum.set_content(val['content'])
        datum.set_source(val['source'])
        datum.set_title(val['title'])
        datum.set_author(val['author'])
        datum.set_url(val['url'])
        datum.set_published(val['published'])
        datum.set_published_utc(val['published_utc'])
        datum.set_collection_utc(val['collection_utc'])
        return datum

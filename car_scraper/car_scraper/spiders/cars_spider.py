import scrapy


class CarSpider(scrapy.Spider):
    name = "cars"

    custom_settings = {
        'AUTOTHROTTLE_ENABLED': True,
    }

    def start_requests(self):
        urls = []
        with open("~/external/urls.txt", "r") as url_file:
            for line in url_file:
                urls.append(line.strip("\n"))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'cars-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)



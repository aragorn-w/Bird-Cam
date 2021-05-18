import os

from bing_image_downloader import downloader


os.chdir("C:\\Users\\wanga\\Desktop\\Personal\\Machine Learning\\Sophomore AI Class\\Bird, Not Bird Data")
search_terms = [
    "close up bird on branch -stock +bird",
    "close up tree branch -bird"
]
for term in search_terms:
    downloader.download(term, limit=1000, output_dir="BingScrapedImages")

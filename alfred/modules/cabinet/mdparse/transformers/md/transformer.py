"""
Images extractor from markdown document.
"""

import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension
from markdown.inlinepatterns import SimpleTagPattern
from typing import List


__all__ = ['ArticleTransformer']


class ImgExtractor(Treeprocessor):
    def run(self, doc):
        """
        Find all images and append to markdown.images.
        """

        self.md.images = []
        for image in doc.findall('.//img'):
            self.md.images.append(image.get('src'))


class ImgExtExtension(Extension):
    def extendMarkdown(self, md, md_globals):
        img_ext = ImgExtractor(md)
        md.treeprocessors.register(img_ext, 'imgext', 20)


class ArticleTransformer:
    """
    Markdown article transformation class.
    """

    def __init__(self, article_path: str, image_downloader):
        self._image_downloader = image_downloader
        self._article_file_path = article_path
        self._md_conv = markdown.Markdown(extensions=[ImgExtExtension(), 'md_in_html'])
        self._replacement_mapping = {}

    def _read_article(self) -> List[str]:
        with open(self._article_file_path, 'r', encoding='utf8') as m_file:
            self._md_conv.convert(m_file.read())

        print(f'Images links count = {len(self._md_conv.images)}')
        images = set(self._md_conv.images)
        print(f'Unique images links count = {len(images)}')

        return images

    def _fix_document_urls(self) -> List[str]:
        # print('Replacing images urls in the document...')
        replacement_mapping = self._replacement_mapping
        lines = []
        with open(self._article_file_path, 'r', encoding='utf8') as infile:
            for line in infile:
                for src, target in replacement_mapping.items():
                    line = line.replace(src, target)
                lines.append(line)

        return lines

    def run(self):
        """
        Run article conversion.
        """

        self._replacement_mapping = self._image_downloader.download_images(self._read_article())
        return self._fix_document_urls()

"""
Images extractor from markdown document.
"""
#
# from lxml import html
# from typing import List
#
#
# __all__ = ['ArticleTransformer']
#
#
# class ImgExtractor:
#     def run(self, doc):
#         """
#         Find all images in HTML.
#         """
#
#         tree = html.fromstring(doc)
#         images = tree.xpath('//img/@src')
#         # links = tree.xpath('//a/@href')
#
#         return images
#
#
#
# class ImgExtExtension(Extension):
#     def extendMarkdown(self, md, md_globals):
#         img_ext = ImgExtractor(md)
#         md.treeprocessors.register(img_ext, 'imgext', 20)
#
#
# class ArticleTransformer:
#     """
#     Markdown article transformation class.
#     """
#
#     def __init__(self, article_path: str, image_downloader):
#         self._image_downloader = image_downloader
#         self._article_file_path = article_path
#         self._md_conv = markdown.Markdown(extensions=[ImgExtExtension()])
#         self._replacement_mapping = {}
#
#     def _read_article(self) -> List[str]:
#         with open(self._article_file_path, 'r') as m_file:
#             self._md_conv.convert(m_file.read())
#
#         print(f'Images links count = {len(self._md_conv.images)}')
#         images = set(self._md_conv.images)
#         print(f'Unique images links count = {len(images)}')
#
#         return images
#
#     def _fix_document_urls(self) -> None:
#         print('Replacing images urls in the document...')
#         replacement_mapping = self._replacement_mapping
#         lines = []
#         with open(self._article_file_path, 'r') as infile:
#             for line in infile:
#                 for src, target in replacement_mapping.items():
#                     line = line.replace(src, target)
#                 lines.append(line)
#
#         with open(self._article_file_path, 'w') as outfile:
#             for line in lines:
#                 outfile.write(line)
#
#     def run(self):
#         """
#         Run article conversion.
#         """
#
#         self._replacement_mapping = self._image_downloader.download_images(self._read_article())
#         self._fix_document_urls()
#

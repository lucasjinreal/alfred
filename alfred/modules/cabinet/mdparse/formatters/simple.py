"""
Simple formatter.
"""


class SimpleFormatter:
    """
    Writes lines, "as is".
    """

    format = 'md'

    @staticmethod
    def write(lines):
        return ''.join(lines).encode('utf8')

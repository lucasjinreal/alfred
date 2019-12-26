"""
Utils using in MANA universe

such as print welcome message
"""
from colorama import Fore, Back, Style


welcome_msg = '''
    __  ______    _   _____    ___    ____
   /  |/  /   |  / | / /   |  /   |  /  _/
  / /|_/ / /| | /  |/ / /| | / /| |  / /  
 / /  / / ___ |/ /|  / ___ |/ ___ |_/ /   
/_/  /_/_/  |_/_/ |_/_/  |_/_/  |_/___/    http://manaai.cn
'''

def welcome(ori_git_url):
    print(Fore.YELLOW + Style.BRIGHT + 'Welcome to MANA AI platform!' + Style.RESET_ALL)
    print(Fore.BLUE + Style.BRIGHT + welcome_msg + Style.RESET_ALL)
    print(Style.BRIGHT + "once you saw this msg, indicates you were back supported by our team!" + Style.RESET_ALL)
    print('the latest updates of our codes always at: {} or {}'.format(ori_git_url, 'http://manaai.cn'))
    print('NOTE: Our codes distributed from anywhere else were not supported!')

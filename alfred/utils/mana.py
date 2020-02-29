#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
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

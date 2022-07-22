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
from loguru import logger
import sys
import time


def init_logger():
    logger.remove()  # Remove the pre-configured handler
    logger.start(
        sys.stderr,
        format="<lvl>{level}</lvl> {time:MM-DD HH:mm:ss} {file}:{line} - {message}",
    )


def formatter(record):
    # package_name = get_package_name()
    filename = record["file"].name
    if len(record['file'].name) > 17:
        filename = record["file"].name[:12] + '..' + record["file"].name[-3:]
    record["extra"].update(filename=filename)
    return "{time:HH:mm:ss MM.DD} <lvl>{level}</lvl> {extra[filename]}:{line}]: {message}\n{exception}"


logger.remove()  # Remove the pre-configured handler
logger.start(
    sys.stderr,
    format=formatter,
)


def save_log_to_file(f_n):
    logger.remove(handler_id=None)  # 清除之前的设置
    # 设置生成日志文件，utf-8编码，每天0点切割，zip压缩，保留3天，异步写入
    logger.add(
        sink=f"{f_n}_{time}.log",
        level="INFO",
        rotation="00:00",
        retention="3 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

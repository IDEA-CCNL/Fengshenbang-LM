# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import datetime
from utils.arguments_parse import args
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

now_date = datetime.datetime.now()
now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')
# 第二步，创建一个handler，用于写入日志文件

if args.log_file_path is None:
    file_handler = logging.FileHandler('./'+str(now_date)+'.log', mode='w')
else:
    file_handler = logging.FileHandler(args.log_file_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)

# 添加handler到logger中
logger.addHandler(file_handler)

# 第三步，创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
logger.addHandler(console_handler)
 


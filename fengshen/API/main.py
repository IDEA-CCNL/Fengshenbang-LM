import uvicorn
import click
import argparse
import json
from importlib import import_module
from fastapi import FastAPI, WebSocket
from starlette.middleware.cors import CORSMiddleware
from utils import user_config, api_logger, setup_logger, RequestDataStructure

# 命令行启动时只输入一个参数，即配置文件的名字，eg: text_classification.json
# 其余所有配置在该配置文件中设定，不在命令行中指定
total_parser = argparse.ArgumentParser("API")
total_parser.add_argument("config_path", type=str)
args = total_parser.parse_args()

# set up user config
user_config.setup_config(args)

# set up logger
setup_logger(api_logger, user_config)

# load pipeline 
pipeline_class = getattr(import_module('fengshen.pipelines.' + user_config.pipeline_type), 'Pipeline')
model_settings = user_config.model_settings
model_args = argparse.Namespace(**model_settings)
pipeline = pipeline_class(
    args = model_args,
    model = user_config.model_name
    )


# initialize app
app = FastAPI(
    title = user_config.PROJECT_NAME, 
    openapi_url = f"{user_config.API_PREFIX_STR}/openapi.json"
    )


# api 
# TODO 
# 需要针对不同请求方法做不同判断，目前仅跑通了较通用的POST方法
# POST方法可以完成大多数 输入文本-返回结果 的请求任务
if(user_config.API_method == "POST"):
    @app.post(user_config.API_path, tags = user_config.API_tags)
    async def fengshen_post(data:RequestDataStructure):
        # logging
        api_logger.info(data.input_text)

        input_text = data.input_text

        result = pipeline(input_text)

        return result
else:
    print("only support POST method")



# Set all CORS enabled origins
if user_config.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins = [str(origin) for origin in user_config.BACKEND_CORS_ORIGINS],
        allow_credentials = user_config.allow_credentials,
        allow_methods = user_config.allow_methods,
        allow_headers = user_config.allow_headers,
    )


if __name__ == '__main__':

    # 启动后可在浏览器打开 host:port/docs 查看接口的具体信息，并可进行简单测试
    # eg: 127.0.0.1:8990/docs
    uvicorn.run(app, host = user_config.SERVER_HOST, port = user_config.SERVER_PORT)
     


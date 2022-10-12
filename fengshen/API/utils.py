from dataclasses import dataclass, field
import os
import json
import logging
from argparse import Namespace
from typing import List, Literal, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, HttpUrl, validator, BaseModel


CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# request body
# 使用pydantic的BaseModel对请求中的body数据进行验证
class RequestDataStructure(BaseModel):
    input_text: List[str] = [""]
    uuid: Optional[int]
    
    # parameters for text2image model
    input_image: Optional[str]
    skip_steps: Optional[int]
    clip_guidance_scale: Optional[int]
    init_scale: Optional[int]

# API config
@dataclass
class APIConfig:

    # server config
    SERVER_HOST: AnyHttpUrl = "127.0.0.1"
    SERVER_PORT: int = 8990
    SERVER_NAME: str = ""
    PROJECT_NAME: str = ""
    API_PREFIX_STR: str = "/api"

    # api config
    API_method: Literal["POST","GET","PUT","OPTIONS","WEBSOCKET","PATCH","DELETE","TRACE","CONNECT"] = "POST"
    API_path: str = "/TextClassification"
    API_tags: List[str] = field(default_factory = lambda: [""])
    
    # CORS config
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = field(default_factory = lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory = lambda: ["*"])
    allow_headers: List[str] = field(default_factory = lambda: ["*"])
    
    # log config
    log_file_path: str = ""
    log_level: str = "INFO"
        
    # pipeline config
    pipeline_type: str = ""
    model_name: str = ""

    # model config
    # device: int = -1
    # texta_name: Optional[str] = "sentence"
    # textb_name: Optional[str] = "sentence2"
    # label_name: Optional[str] = "label"
    # max_length: int = 512
    # return_tensors: str = "pt"
    # padding: str = "longest"
    # truncation: bool = True
    # skip_special_tokens: bool = True
    # clean_up_tkenization_spaces: bool = True

    # # parameters for text2image model
    # skip_steps: Optional[int] = 0
    # clip_guidance_scale: Optional[int] = 0
    # init_scale: Optional[int] = 0

    def setup_config(self, args:Namespace) -> None:
        
        # load config file
        with open(CURRENT_DIR_PATH + "/config/" + args.config_path, "r") as jsonfile:
            config = json.load(jsonfile)

        server_config = config["SERVER"]
        logging_config = config["LOGGING"]
        pipeline_config = config["PIPELINE"]
        
        # server config
        self.SERVER_HOST: AnyHttpUrl = server_config["SERVER_HOST"]
        self.SERVER_PORT: int = server_config["SERVER_PORT"]
        self.SERVER_NAME: str = server_config["SERVER_NAME"]
        self.PROJECT_NAME: str = server_config["PROJECT_NAME"]
        self.API_PREFIX_STR: str = server_config["API_PREFIX_STR"]

        # api config
        self.API_method: Literal["POST","GET","PUT","OPTIONS","WEBSOCKET","PATCH","DELETE","TRACE","CONNECT"] = server_config["API_method"]
        self.API_path: str = server_config["API_path"]
        self.API_tags: List[str] = server_config["API_tags"]

        # CORS config
        self.BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = server_config["BACKEND_CORS_ORIGINS"]
        self.allow_credentials: bool = server_config["allow_credentials"]
        self.allow_methods: List[str] = server_config["allow_methods"]
        self.allow_headers: List[str] = server_config["allow_headers"]
        
        # log config
        self.log_file_path: str = logging_config["log_file_path"]
        self.log_level: str = logging_config["log_level"]
        
        # pipeline config
        self.pipeline_type: str = pipeline_config["pipeline_type"]
        self.model_name: str = pipeline_config["model_name"]

        # general model config
        self.model_settings: dict = pipeline_config["model_settings"]

        # 由于pipeline本身会解析参数，后续参数可以不要
        # 直接将model_settings字典转为Namespace后作为pipeline的args参数即可

        # self.device: int = self.model_settings["device"]
        # self.texta_name: Optional[str] = self.model_settings["texta_name"]
        # self.textb_name: Optional[str] = self.model_settings["textb_name"]
        # self.label_name: Optional[str] = self.model_settings["label_name"]
        # self.max_length: int = self.model_settings["max_length"]
        # self.return_tensors: str = self.model_settings["return_tensors"]
        # self.padding: str = self.model_settings["padding"]
        # self.truncation: bool = self.model_settings["truncation"]
        # self.skip_special_tokens: bool = self.model_settings["skip_special_tokens"]
        # self.clean_up_tkenization_spaces: bool = self.model_settings["clean_up_tkenization_spaces"]

        # # specific parameters for text2image model
        # self.skip_steps: Optional[int] = self.model_settings["skip_steps"]
        # self.clip_guidance_scale: Optional[int] = self.model_settings["clip_guidance_scale"]
        # self.init_scale: Optional[int] = self.model_settings["init_scale"]
        


def setup_logger(logger, user_config: APIConfig):
        
        # default level: INFO 

        logger.setLevel(getattr(logging, user_config.log_level, "INFO"))
        ch = logging.StreamHandler()
        
        if(user_config.log_file_path == ""):
            fh = logging.FileHandler(filename = CURRENT_DIR_PATH + "/log/"  + user_config.SERVER_NAME  + ".log")
        elif(".log" not in user_config.log_file_path[-5:-1]):
            fh = logging.FileHandler(filename = user_config.log_file_path + "/" + user_config.SERVER_NAME + ".log")
        else:
            fh = logging.FileHandler(filename = user_config.log_file_path)


        formatter = logging.Formatter(
            "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
        )

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)  # Exporting logs to the screen
        logger.addHandler(fh)  # Exporting logs to a file

        return logger

user_config = APIConfig()
api_logger = logging.getLogger()










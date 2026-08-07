from ai_research_assistant.utils.common import *
from ai_research_assistant.constants import *

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):     # Access to constants

        self.config = read_yaml(config_filepath) # read all config and params yaml files

        create_directories([self.config.artifacts_root])
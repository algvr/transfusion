from envyaml import EnvYAML


class EnvYAMLWrapper(EnvYAML):
    def __init__(self, yaml_file=None, env_file=None, include_environment=True, strict=True, flatten=True, **kwargs):
        super().__init__(yaml_file, env_file, include_environment, strict, flatten, **kwargs)
        # read yaml file and parse it
        yaml_config = self._EnvYAML__read_yaml_file(
            self._EnvYAML__get_file_path(yaml_file, "ENV_YAML_FILE", self.DEFAULT_ENV_YAML_FILE),
            self._EnvYAML__cfg,
            self._EnvYAML__strict,
        )
        self.yaml_config = yaml_config
    

    def set(self, key, value):
        self._EnvYAML__cfg[key] = value
        self.yaml_config[key] = value

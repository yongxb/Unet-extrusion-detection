import glob
import os
from ruamel.yaml import YAML
from tensorpack import logger

yaml = YAML()


######################
# Accessors/Mutators
######################
def get_parameter(config, parameter):
    """Output the value from the config file using the given key

    Parameters
    ----------
    parameter : `list` or `str`
        Key or list of keys used to find for the value in the config file

    config : `list`, optional
        Used to iterate through nested dictionaries. Required to recursively iterate through neseted dictionary

    Returns
    ----------
    value : `str` or `int` or `list`
        Value obtained from the specified key

    See Also
    ----------
    find_key : Function to identify the list of keys to address the correct item in a nested dictionary
    """
    assert isinstance(parameter, (list, str))

    # find for key in nested dictionary
    if isinstance(parameter, str):
        temp_parameter = parameter
        parameter = find_key(config, parameter)
        # return None for empty parameter
        if parameter is None:
            logger.warn("Parameter '{}' is not found in config file. Returning None.".format(temp_parameter))
            return None

    if not parameter:
        return config

    return get_parameter(config.get(parameter[0]), parameter[1:])


def find_key(config, key):
    """Find the list of keys to address the correct item in a nested dictionary

    Parameters
    ----------
    key : `str`
        Key that needs to be correctly addressed in a nested dictionary

    config : `list` or `none`, optional
        Used to iterate through nested dictionaries

    Returns
    ----------
    key : `list`
        Address of the key in the nested dictionary
    """
    for k, v in config.items():
        if k == key:
            return [k]
        elif isinstance(v, dict):
            found_key = find_key(config=v, key=key)
            if found_key is not None:
                return [k] + found_key


def update_parameter(config, parameter, value):
    """Updates the parameter in the config file using a full addressed list

    Parameters
    ----------
    parameter : `list`
        List of keys that point to the correct item in the nested dictionary

    value : `str` or `int` or `list`
        Value that is updated in the nested dictionary

    config : `list` or `none`, optional
        Used to iterate through nested dictionaries

    Returns
    ----------
    TODO
    """
    assert type(parameter) is list

    if config is None:
        logger.error("Error in update_parameter")

    if len(parameter) == 1:
        config.update({parameter[0]: value})
        return config
    return update_parameter(config.get(parameter[0]), parameter[1:], value)


######################
# Config IO options
######################
def load_config_from_file(file_path):
    """Load parameters from yaml file

    Parameters
    ----------
    file_path : `str`
        Path of config file to load

    Returns
    ----------
    config : `dict`
        Dictionary containing the config parameters
    """

    with open(file_path, 'r') as input_file:
        config = yaml.load(input_file)
    input_file.close()

    return config


def load_config_from_model_dir(model_dir):
    """Finds for a config file from the model directory and loads it

    Parameters
    ----------
    model_dir : `str`
        Folder to search for and load the config file

    Returns
    ----------
    config : `dict`
        Dictionary containing the config parameters

    Raises
    ------
    IndexError
        If there are no config file in the model_dir
    """

    # check if yaml file exists in model_dir
    try:
        list_config_files = glob.glob(os.path.join(model_dir, '*config.yml'))
        if len(list_config_files) > 1:
            logger.warn("Multiple config files found. Loading {}".format(list_config_files[0]))
        else:
            logger.info("Config file exists in model directory. Loading {}".format(list_config_files[0]))
        return load_config_from_file(list_config_files[0])
    except IndexError:
        logger.error("No config file found in model_dir.")
        raise


def write_config(config, file_path):
    """Writes parameters to yaml file

    Parameters
    ----------
    file_path : `str`
        Path of config file to write to
    """

    with open(file_path, 'w') as output_file:
        yaml.dump(config, output_file)

    output_file.close()

    logger.info("Config file written to: {}".format(file_path))


def write_model(model, file_path):
    """Writes parameters to yaml file

    Parameters
    ----------
    model : :class:`Keras.model`
        Keras model that will be parsed and written to a yaml file

    file_path : `str`
        Path of model file to write to
    """

    with open(file_path, 'w') as output_file:
        output_file.write(model.to_yaml())

    output_file.close()

    logger.info("Model file written to: {}".format(file_path))

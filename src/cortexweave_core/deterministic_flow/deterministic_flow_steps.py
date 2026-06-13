
import logging
import os

import yaml

logger = logging.getLogger(__name__)

# workflow steps dictionary of all agents.
# The dictionary has "step_name" as the key and "step_content" as the value.
# "step_content" contains prompt to be played, instructions to be executed etc.
# THIS VARIABLE MAY BE REFERRED IN THE TOOLS/MODULES OF ALL AGENTS
WORKFLOW_STEPS = {}

# def add_workflow_step(key, value):
#     global WORKFLOW_STEPS
#     if key in WORKFLOW_STEPS:
#         logger.warning(f"Duplicate step '{key}' found, overwriting.")
#     WORKFLOW_STEPS[key] = value
#     logger.info(f"*****  added workflow step {key}:  Value: {WORKFLOW_STEPS[key]}")

async def get_workflow_step(key):
    global WORKFLOW_STEPS
    dict_keys = WORKFLOW_STEPS.keys()
    logger.info(f"*****  Keys of a WORKFLOW_STEPS object: {dict_keys}")
    if key in WORKFLOW_STEPS:
        return WORKFLOW_STEPS[key]
    return None

def load_workflow_yaml():
    global WORKFLOW_STEPS
    project_working_directory = os.getcwd()
    logger.info(f"project_working_directory: {project_working_directory}")
    workflow_dirs = _find_workflow_folders(project_working_directory)

    for workflow_dir in workflow_dirs:
        for filename in os.listdir( workflow_dir):
            if not filename.endswith(".yaml") and not filename.endswith(".yml"):
                continue

            filepath = os.path.join(workflow_dir, filename)
            logger.info(f"Reading workflow steps yaml file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    data = yaml.safe_load(file)
                    if data and "workflow" in data:
                        for step_entry in data["workflow"]:
                            step_name = step_entry.get("step")
                            if not step_name:
                                logger.warning(f"Missing 'step' key in one of the entries in {filepath}")
                                continue
                            if step_name in WORKFLOW_STEPS:
                                logger.warning(f"Duplicate step '{step_name}' found in {filepath}, overwriting.")
                            WORKFLOW_STEPS[step_name] = step_entry
                    else:
                        logger.warning(f"Missing 'workflow' element in {filepath}. Is this a workflow steps yaml file?")
                except Exception as e:
                    logger.error(f"Failed to load workflow steps yaml file: {filepath}: {e}")

def _find_workflow_folders(root_dir):
    """
    Finds all folders named 'workflow' within a given root directory
    and its nested subdirectories.

    Args:
        root_dir (str): The starting directory to search from.

    Returns:
        list: A list of full paths to all 'workflow' folders found.
    """
    workflow_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'workflow' in dirnames:
            # Construct the full path to the 'workflow' folder
            workflow_folder_path = os.path.join(dirpath, 'workflow')
            workflow_folders.append(workflow_folder_path)
    return workflow_folders

# load workflow steps
load_workflow_yaml()

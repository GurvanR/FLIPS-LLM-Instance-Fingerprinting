# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Folder creation and repository-root discovery."""

import os


def make_path_and_create_folder(save_path: str, model_name: str) -> str:
    """Create a model folder under save_path, handling 'org/model' names."""
    splitted = model_name.split("/")
    folder_path = os.path.join(save_path, model_name)
    if len(splitted) == 2:
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    if len(splitted) > 2:
        raise ValueError(" The name of the model contains more than one '/' ")

    return folder_path


def get_repository_level_path() -> str:
    """Root of the project: the first directory above this file holding ``pyproject.toml``."""
    repository_level = os.path.abspath(__file__)

    while not _is_project_root(repository_level) and repository_level != os.path.dirname(repository_level):
        repository_level = os.path.dirname(repository_level)

    return repository_level


def _is_project_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "pyproject.toml"))

"""
Package de preprocessing pour le projet Pomme de Pain ML.
"""

__version__ = "0.1.0"
__author__ = "Votre Nom"

from .config import data_config, processing_config, logging_config, create_directories
from .utils import setup_logger, load_file, save_file, print_summary

# Créer les répertoires au moment de l'import
create_directories()
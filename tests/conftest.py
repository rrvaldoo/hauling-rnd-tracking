from pathlib import Path

import pytest

from hauling.config import read_config

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG = """[CONFIG]
resolution_area = 100,100
main_area_coordinates_list = [[0,0],[100,0],[100,100],[0,100]]
speed_area_start_coordinates_list = [[0,10],[100,10],[100,20],[0,20]]
speed_area_final_coordinates_list = [[0,40],[100,40],[100,50],[0,50]]
final_area_coordinates_list = [[0,60],[100,60],[100,70],[0,70]]
distance_area_speed = [47]
speed_limit = 60
travel_direction = South
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "camera.conf"
    path.write_text(SYNTHETIC_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def config(config_path):
    return read_config(config_path)

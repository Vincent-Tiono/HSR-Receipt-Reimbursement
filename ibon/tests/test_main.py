import pytest
import json
import os
from ibon.main import extract

# Parameterized test cases
@pytest.mark.parametrize(
    "img_path, expected_output",
    [
        ("ibon/images/1.jpg", {
            "date": "2017/01/28",
            "price": 1000,
            "dep_station": "Taipei",
            "arr_station": "Taichung",
            "serial_number": "2104810266181",
        }),
        ("ibon/images/2.jpg", {
            "date": "2020/08/02",
            "price": 950,
            "dep_station": "Chiayi",
            "arr_station": "Taipei",
            "serial_number": "2101502134115",
        }),
        ("ibon/images/3.jpg", {
            "date": "2022/12/04",
            "price": 630,
            "dep_station": "Taipei",
            "arr_station": "Taichung",
            "serial_number": "2104003275532",
        }),
        ("ibon/images/4.jpg", {
            "date": "2024/09/21",
            "price": 1350,
            "dep_station": "Tainan",
            "arr_station": "Taipei",
            "serial_number": "2100802494810",
        }),
        ("ibon/images/5.jpg", {
            "date": "2019/09/07",
            "price": 890,
            "dep_station": "Tainan",
            "arr_station": "Taoyuan",
            "serial_number": "2104412472375",
        }),
        ("ibon/images/6.jpg", {
            "date": "2018/04/26",
            "price": 900,
            "dep_station": "Tainan",
            "arr_station": "Nangang",
            "serial_number": "2105900963218",
        }),
        ("ibon/images/7.jpg", {
            "date": "2020/08/27",
            "price": 1080,
            "dep_station": "Taipei",
            "arr_station": "Tainan",
            "serial_number": "2105302396447",
        }),
        ("ibon/images/8.jpg", {
            "date": "2017/12/03",
            "price": 1350,
            "dep_station": "Taipei",
            "arr_station": "Tainan",
            "serial_number": "2103913153283",
        }),
        ("ibon/images/9.jpg", {
            "date": "2017/05/28",
            "price": 570,
            "dep_station": "Taipei",
            "arr_station": "Changhua",
            "serial_number": "2103611265802",
        }),
        ("ibon/images/10.jpg", {
            "date": "2024/01/04",
            "price": 1190,
            "dep_station": "Tainan",
            "arr_station": "Taoyuan",
            "serial_number": "2101000020171",
        }),
    ],
)

def test_extract(img_path, expected_output):
    # First check if the image exists
    assert os.path.exists(img_path), f"Test image not found: {img_path}"
    
    # Process the image to get bytes - disable visual display for tests
    # processed_image_bytes = process_image(img_path)
    
    # Execute the extraction function with the processed image bytes
    result = extract(img_path)
    result_dict = json.loads(result)

    # Debug output
    print(f"\nTesting {img_path}")
    print(f"Expected: {expected_output}")
    print(f"Actual:   {result_dict}")

    # Validate the extracted values
    assert isinstance(result_dict, dict), "Function did not return a dictionary"
    assert result_dict["date"] == expected_output["date"], \
        f"Date mismatch: {result_dict['date']} != {expected_output['date']}"
    assert result_dict["price"] == expected_output["price"], \
        f"Price mismatch: {result_dict['price']} != {expected_output['price']}"
    assert result_dict["dep_station"] == expected_output["dep_station"], \
        f"Departure station mismatch: {result_dict['dep_station']} != {expected_output['dep_station']}"
    assert result_dict["arr_station"] == expected_output["arr_station"], \
        f"Arrival station mismatch: {result_dict['arr_station']} != {expected_output['arr_station']}"
    assert result_dict["serial_number"] == expected_output["serial_number"], \
        f"Serial mismatch: {result_dict['serial_number']} != {expected_output['serial_number']}"
import pytest
import json
from card.main_without_saving import extract

# Parameterized test cases
@pytest.mark.parametrize(
    "img_path, expected_output",
    [
        ("card/images/1.jpg", {
            "date": "2024/08/13",
            "price": 395,
            "dep_station": "Zuoying",
            "arr_station": "Taichung",
            "serial_number": "1221202260145",
        }),
        ("card/images/2.jpg", {
            "date": "2024/11/11",
            "price": 700,
            "dep_station": "Taichung",
            "arr_station": "Taipei",
            "serial_number": "0711203160208",
        }),
        ("card/images/3.jpg", {
            "date": "2023/07/03",
            "price": 1145,
            "dep_station": "Taipei",
            "arr_station": "Tainan",
            "serial_number": "0224211840583",
        }),
        ("card/images/4.jpg", {
            "date": "2021/05/03",
            "price": 540,
            "dep_station": "Zuoying",
            "arr_station": "Yunlin",
            "serial_number": "1213111230106",
        }),
        ("card/images/5.jpg", {
            "date": "2023/07/03",
            "price": 1145,
            "dep_station": "Taipei",
            "arr_station": "Tainan",
            "serial_number": "0224211840583",
        }),
        ("card/images/6.jpg", {
            "date": "2023/11/19",
            "price": 375,
            "dep_station": "Miaoli",
            "arr_station": "Taipei",
            "serial_number": "0620313230261",
        }),
        ("card/images/7.jpg", {
            "date": "2024/12/20",
            "price": 215,
            "dep_station": "Taipei",
            "arr_station": "Miaoli",
            "serial_number": "0220503550373",
        }),
        ("card/images/8.jpg", {
            "date": "2024/07/05",
            "price": 1045,
            "dep_station": "Chiayi",
            "arr_station": "Taipei",
            "serial_number": "1020701870009",
        }),
        ("card/images/9.jpg", {
            "date": "2025/01/12",
            "price": 700,
            "dep_station": "Taichung",
            "arr_station": "Taipei",
            "serial_number": "0222003570563",
        }),
        ("card/images/10.jpg", {
            "date": "2022/02/16",
            "price": 700,
            "dep_station": "Taipei",
            "arr_station": "Taichung",
            "serial_number": "0210300470119",
        })
    ],
)
def test_extract(img_path, expected_output):
    # Execute the extraction function and parse the JSON output
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
        f"dep_station mismatch: {result_dict['dep_station']} != {expected_output['dep_station']}"
    assert result_dict["arr_station"] == expected_output["arr_station"], \
        f"arr_station mismatch: {result_dict['arr_station']} != {expected_output['arr_station']}"
    assert result_dict["serial_number"] == expected_output["serial_number"], \
        f"Serial mismatch: {result_dict['serial_number']} != {expected_output['serial_number']}"
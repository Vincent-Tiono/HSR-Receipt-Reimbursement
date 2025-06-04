import pytest
import json
from fami.main import extract

# Parameterized test cases
@pytest.mark.parametrize(
    "img_path, expected_output",
    [
        ("fami/images/1.jpg", {
            "date": "2019/03/28",
            "price": 395,
            "dep_station": "Taichung",
            "arr_station": "Zuoying",
            "serial_number": "2202110870391",
        }),
        ("fami/images/2.jpg", {
            "date": "2023/05/22",
            "price": 525,
            "dep_station": "Taipei",
            "arr_station": "Taichung",
            "serial_number": "2200611416411",
        }),
        ("fami/images/3.jpg", {
            "date": "2025/02/22",
            "price": 1350,
            "dep_station": "Taipei",
            "arr_station": "Tainan",
            "serial_number": "2203510516789",
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
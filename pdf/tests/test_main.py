import pytest
import json
from pdf.main import extract

@pytest.mark.parametrize(
    "img_path, expected_output",
    [
        ("pdf/docs/pdf_1.pdf", {"date": "2024/12/21", "price": 675, "dep_station": "Taipei", "arr_station": "Taichung", "serial_number": "2971703566301"}),
        ("pdf/docs/pdf_2.pdf", {"date": "2024/08/04", "price": 675, "dep_station": "Taichung", "arr_station": "Taipei", "serial_number": "2973802174369"}),
        ("pdf/docs/pdf_3.pdf", {"date": "2024/08/23", "price": 725, "dep_station": "Nangang", "arr_station": "Taichung", "serial_number": "2975802363749"}),
        ("pdf/docs/pdf_4.pdf", {"date": "2024/08/25", "price": 675, "dep_station": "Taichung", "arr_station": "Taipei", "serial_number": "2976102380935"}),
        ("pdf/docs/pdf_5.pdf", {"date": "2024/11/22", "price": 725, "dep_station": "Nangang", "arr_station": "Taichung", "serial_number": "2974103275029"}),
        ("pdf/docs/pdf_6.pdf", {"date": "2024/12/22", "price": 600, "dep_station": "Taipei", "arr_station": "Taichung", "serial_number": "2900503572160"}),
        ("pdf/docs/pdf_7.pdf", {"date": "2024/11/25", "price": 1250, "dep_station": "Taichung", "arr_station": "Taipei", "serial_number": "2903403300091"}),
    ],
)

# @pytest.mark.parametrize(
#     "img_path, expected_output",
#     [
#         ("pdf/docs/pdf_1.pdf", {"date": "2024/12/21", "price": 675, "dep_station": "台北", "arr_station": "台中", "serial_number": "2971703566301"}),
#         ("pdf/docs/pdf_2.pdf", {"date": "2024/08/04", "price": 675, "dep_station": "台中", "arr_station": "台北", "serial_number": "2973802174369"}),
#         ("pdf/docs/pdf_3.pdf", {"date": "2024/08/23", "price": 725, "dep_station": "南港", "arr_station": "台中", "serial_number": "2975802363749"}),
#         ("pdf/docs/pdf_4.pdf", {"date": "2024/08/25", "price": 675, "dep_station": "台中", "arr_station": "台北", "serial_number": "2976102380935"}),
#         ("pdf/docs/pdf_5.pdf", {"date": "2024/11/22", "price": 725, "dep_station": "南港", "arr_station": "台中", "serial_number": "2974103275029"}),
#         ("pdf/docs/pdf_6.pdf", {"date": "2024/12/22", "price": 600, "dep_station": "台北", "arr_station": "台中", "serial_number": "2900503572160"}),
#         ("pdf/docs/pdf_7.pdf", {"date": "2024/11/25", "price": 1250, "dep_station": "台中", "arr_station": "台北", "serial_number": "2903403300091"}),
#     ],
# )

def test_extract(img_path, expected_output):
    result = extract(img_path)  # Call the actual function

    # Parse the string result into a dictionary
    result_dict = json.loads(result)  # Convert the JSON string into a dictionary

    print(f"\nTesting {img_path}")
    print(f"Expected: {expected_output}")
    print(f"Actual:   {result_dict}")

    # Assertions
    assert isinstance(result_dict, dict), "Function did not return a dictionary"
    assert result_dict["date"] == expected_output["date"], f"Date mismatch: {result_dict['date']} != {expected_output['date']}"
    assert result_dict["price"] == expected_output["price"], f"Price mismatch: {result_dict['price']} != {expected_output['price']}"
    assert result_dict["dep_station"] == expected_output["dep_station"], f"Departure station mismatch: {result_dict['dep_station']} != {expected_output['dep_station']}"
    assert result_dict["arr_station"] == expected_output["arr_station"], f"Arrival station mismatch: {result_dict['arr_station']} != {expected_output['arr_station']}"
    assert result_dict["serial_number"] == expected_output["serial_number"], f"Serial mismatch: {result_dict['serial_number']} != {expected_output['serial_number']}"
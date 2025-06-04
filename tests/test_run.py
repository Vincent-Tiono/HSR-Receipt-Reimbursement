import pytest
import json
from run import extract

@pytest.mark.parametrize(
    "img_path, expected_output",
    [
        ("card/images/image_1.png", {"date": "2018/10/24", "price": 1530, "serial_number": "0226202970006"}),
        ("card/images/image_2.png", {"date": "2024/02/25", "price": 1445, "serial_number": "0221000560739"}),
        ("card/images/image_3.png", {"date": "2023/01/09", "price": 280, "serial_number": "0210310090200"}),
        ("card/images/image_4.png", {"date": "2024/02/25", "price": 1445, "serial_number": "0221000560739"}),
        ("card/images/image_5.png", {"date": "2020/12/10", "price": 1390, "serial_number": "0721203450332"}),
        ("fami/images/image_1_fami.png", {"date": "2021/05/17", "price": 595, "serial_number": "2200511317525"}),
        ("fami/images/image_2_fami.png", {"date": "2021/05/12", "price": 40, "serial_number": "2203111123676"}),
        ("fami/images/image_3_fami.png", {"date": "2019/12/28", "price": 700, "serial_number": "2202913423616"}),
        ("fami/images/image_4_fami.png", {"date": "2024/10/10", "price": 150, "serial_number": "2203702762562"}),
        ("fami/images/image_5_fami.png", {"date": "2023/11/07", "price": 560, "serial_number": "2201613106358"}),
        ("ibon/images/image_1_ibon.png", {"date": "2021/05/12", "price": 30, "serial_number": "2104711122469"}),
        ("ibon/images/image_2_ibon.png", {"date": "2024/03/15", "price": 350, "serial_number": "2105100742279"}),
        ("ibon/images/image_3_ibon.png", {"date": "2022/12/04", "price": 630, "serial_number": "2104003275532"}),
        ("ibon/images/image_4_ibon.png", {"date": "2018/04/04", "price": 1120, "serial_number": "2105400945682"}),
        ("ibon/images/image_5_ibon.png", {"date": "2019/11/25", "price": 700, "serial_number": "2105113282290"}),
        ("ibon/images/image_6_ibon.png", {"date": "2024/03/22", "price": 745, "serial_number": "2105100743717"}),
        ("ibon/images/image_7_ibon.png", {"date": "2017/01/28", "price": 1000, "serial_number": "2104810266181"}),
        ("ibon/images/image_8_ibon.png", {"date": "2019/09/14", "price": 820, "serial_number": "2100112563763"}),
        ("ibon/images/image_9_ibon.png", {"date": "2023/11/22", "price": 1200, "serial_number": "2101013197872"}),
        ("ibon/images/image_10_ibon.png", {"date": "2023/07/02", "price": 1330, "serial_number": "2103211837571"}),
    ],
)
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
    assert result_dict["serial_number"] == expected_output["serial_number"], f"Serial mismatch: {result_dict['serial_number']} != {expected_output['serial_number']}"

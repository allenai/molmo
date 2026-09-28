import numpy as np

from olmo.util import extract_points


def test_click_is_not_parsed_again_as_a_plain_point():
    points = extract_points("Click(10.2, 20.5) (30.0, 40.0)", 100, 100)
    assert len(points) == 2
    np.testing.assert_allclose(points[0], [10.2, 20.5])
    np.testing.assert_allclose(points[1], [30.0, 40.0])


def test_plain_point_still_parses():
    points = extract_points("(10.2, 20.5)", 100, 100)
    assert len(points) == 1
    np.testing.assert_allclose(points[0], [10.2, 20.5])

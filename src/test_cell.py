"""
GridCell tests
"""

from cell import STRICT_CELL_ARITHMETIC, GridCell


def test_add():
    for c in GridCell:
        assert c + GridCell.EMPTY == c
        assert GridCell.EMPTY + c == c

    if STRICT_CELL_ARITHMETIC:
        for c in GridCell:
            for d in GridCell:
                if c != GridCell.EMPTY and d != GridCell.EMPTY:
                    try:
                        _ = c + d
                        assert (
                            False
                        ), f"Adding {repr(c)} and {repr(d)} should raise an error"
                    except ValueError:
                        pass
    else:
        for c in GridCell:
            for d in GridCell:
                if c != GridCell.EMPTY and d != GridCell.EMPTY:
                    assert c + d == GridCell.EMPTY


def test_sub():
    for c in GridCell:
        assert c - GridCell.EMPTY == c
        for d in GridCell:
            if c == d:
                assert c - d == GridCell.EMPTY
            elif d != GridCell.EMPTY:
                if STRICT_CELL_ARITHMETIC:
                    try:
                        _ = c - d
                        assert (
                            False
                        ), f"Subtracting {repr(d)} from {repr(c)} should raise an error"
                    except ValueError:
                        pass
                else:
                    assert c - d == GridCell.EMPTY


def test_bitwise_operations():
    # EMPTY = 0b0001
    # RED = 0b0010
    # BLUE = 0b0100
    # NEUTRAL = 0b1000

    assert GridCell.RED & GridCell.BLUE == 0
    assert GridCell.RED & GridCell.EMPTY == 0
    assert GridCell.RED | GridCell.BLUE == 6
    assert GridCell.RED | GridCell.EMPTY == 3
    assert GridCell.RED ^ GridCell.BLUE == 6
    assert GridCell.RED ^ GridCell.EMPTY == 3
    assert GridCell.RED ^ GridCell.RED == 0
    assert ~GridCell.RED == -3
    assert ~GridCell.EMPTY == -2
    assert ~GridCell.BLUE == -5
    assert ~GridCell.NEUTRAL == -9
    assert GridCell.RED << 1 == 4
    assert GridCell.RED >> 1 == 1
    assert GridCell.BLUE << 1 == 8
    assert GridCell.BLUE >> 1 == 2
    assert GridCell.EMPTY << 1 == 2
    assert GridCell.EMPTY >> 1 == 0

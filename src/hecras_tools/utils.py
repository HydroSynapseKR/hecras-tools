from ast import literal_eval

CROSS_SECTION_RENAME_MAP = {
    "River": "river",
    "Reach": "reach",
    "RS": "station",
    "Description": "description",
    "Len Left": "reach_len_left",
    "Len Channel": "reach_len_chan",
    "Len Right": "reach_len_right",
    "Left Bank": "bank_sta_left",
    "Right Bank": "bank_sta_right",
    "Friction Mode": "n_mode",
    "Contr": "contr_coeff",
    "Expan": "expan_coeff",
    "HP Count": "hp_count",
    "HP Start Elev": "hp_start_el",
    "HP Vert Incr": "hp_ver_incr",
    "HP LOB Slices": "hp_lob_slices",
    "HP Chan Slices": "hp_chan_slices",
    "HP ROB Slices": "hp_rob_slices",
    "Default Centerline": "default_centerline",
    "Skew": "skew",
    "PC Invert": "pc_invert",
    "PC Width": "pc_width",
    "PC Mann": "pc_mann",
    "Deck Preissman Slot": "deck_preissman_slot",
    "Contr (USF)": "contr_coeff_usf",
    "Expan (USF)": "expan_coeff_usf"
}


def safe_literal_eval(val):
    """Safely evaluate strings that might represent Python literals.
    This is typically used to normalize river station data so lookup can be performed safely.
    values are first rounded to 5 places to resolve read errors
    sometimes values like 13 are read as 12.9999999 or 13.0000001
    """
    try:
        val_c = literal_eval(val)
        if isinstance(val_c, float) and round(val_c, 5).is_integer():
            return int(val_c)
        else:
            return val_c
    except (ValueError, SyntaxError):
        if isinstance(val, float) and round(val, 5).is_integer():
            return int(val)
        else:
            return val


def coordinate_lines_to_list(coord_lines: list, chars: int = 16) -> list[tuple]:
    """
    Convert a coordinate string into a list of (x,y) tuples.
    Args:
        coord_lines: list of strings representing each line of coordinates
        chars: int representing how many spaces each x and y coordinate occupy

    Returns:
        list of (x, y) tuples
    """
    cleaned_str = [s.strip('\n') for s in coord_lines]
    pairs = [item for s in cleaned_str for item in zip(*(iter([s[i:i + chars] for i in range(0, len(s), chars)]),) * 2)]
    return pairs


def coordinate_list_to_lines(xy_lst: list[tuple], item_per_line: int, chars: int) -> list:
    """
    Convert a list of (x,y) tuples into a list of strings in the hec-ras geometry format.
    Args:
        xy_lst: list of (x, y) coordinates
        item_per_line: int for how many coordinates are included in a single line
        chars: int for how much space each x and y coordinates take
    Returns:
        list of lines that can be written to geometry text file
    """
    lines = [''.join([f"{x: >{chars}.2f}{y: >{chars}.2f}" for x, y in xy_lst[i:i + item_per_line]]) + '\n' for i in
             range(0, len(xy_lst), item_per_line)]
    return [x.replace('nan', '   ') for x in lines]


def lines_to_list(lines, chars=16) -> list:
    """
    Convert a list of (x,y) tuples into a list of strings in the hec-ras geometry format.
    Args:
        lines: list of strings
        chars: int representing how much space each item take
    Returns:
        list: list of data
    """
    ln_str = [s.strip('\n') for s in lines]
    return [float(item) for s in ln_str for item in [s[i:i + chars] for i in range(0, len(s), chars)]]


def list_to_lines(data_list, item_per_line, chars) -> list:
    """
    Convert a list of items into a list of strings in the hec-ras geometry format.
    Args:
        data_list: list of data
        item_per_line: int representing how many items are included in a single line
        chars: int representing how much space each item takes
    Returns:
        list of lines that can be written to geometry text file
    """
    return_lines = [''.join([f"{x: >{chars}.2f}" for x in data_list[i:i + item_per_line]]) + '\n' for i in
                    range(0, len(data_list), item_per_line)]
    cleaned = [x.replace('nan', '   ') for x in return_lines]
    return cleaned

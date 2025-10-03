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

STRUCTURE_RENAME_MAP = {
    "Groupname": "structure_key",
    "Type": "structure_type",
    "Mode": "structure_mode",
    "River": "river",
    "Reach": "reach",
    "RS": "station",
    "Connection": "connection",
    "US Type": "us_type",
    "US River": "us_river",
    "US Reach": "us_reach",
    "US RS": "us_station",
    "US SA/2D": "us_sa2d",
    "DS Type": "ds_type",
    "DS River": "ds_river",
    "DS Reach": "ds_reach",
    "DS RS": "ds_station",
    "DS SA/2D": "ds_sa2d",
    "Node Name": "node_name",
    "Description": "description",
    "Last Edited": "last_edited",
    "Upstream Distance": "upstream_distance",
    "Weir Width": "weir_width",
    "Weir Max Submergence": "weir_max_submergence",
    "Weir Min Elevation": "weir_min_elevation",
    "Weir Coef": "weir_coefficient",
    "Weir Shape": "weir_shape",
    "Weir Design EG Head": "weir_design_eg_head",
    "Weir Design Spillway HT": "weir_design_spillway_ht",
    "Weir US Slope": "weir_us_slope",
    "Weir DS Slope": "weir_ds_slope",
    "Linear Routing Positive Coef": "linear_routing_positive_coef",
    "Linear Routing Negative Coef": "linear_routing_negative_coef",
    "Linear Routing Elevation": "linear_routing_elevation",
    "LW HW Position": "lw_hw_position",
    "LW TW Position": "lw_tw_position",
    "LW HW Distance": "lw_hw_distance",
    "LW TW Distance": "lw_tw_distance",
    "LW Span Multiple": "lw_span_multiple",
    "Use 2D for Overflow": "use_2d_for_overflow",
    "Use Velocity into 2D": "use_velocity_into_2d",
    "Hagers Weir Coef": "hagers_weir_coefficient",
    "Hagers Height": "hagers_height",
    "Hagers Slope": "hagers_slope",
    "Hagers Angle": "hagers_angle",
    "Hagers Radius": "hagers_radius",
    "Use WS for Weir Reference": "use_ws_for_weir_reference",
    "Pilot Flow": "pilot_flow",
    "Culvert Groups": "culvert_groups",
    "Culverts Flap Gates": "culverts_flap_gates",
    "Gate Groups": "gate_groups",
    "HTAB FF Points": "htab_ff_points",
    "HTAB RC Count": "htab_rc_count",
    "HTAB RC Points": "htab_rc_points",
    "HTAB HW Max": "htab_hw_max",
    "HTAB TW Max": "htab_tw_max",
    "HTAB Max Flow": "htab_max_flow",
    "Cell Spacing Near": "cell_spacing_near",
    "Cell Spacing Far": "cell_spacing_far",
    "Near Repeats": "near_repeats",
    "Protection Radius": "protection_radius",
    "Use Friction in Momentum": "use_friction_in_momentum",
    "Use Weight in Momentum": "use_weight_in_momentum",
    "Use Critical US": "use_critical_us",
    "Use EG for Pressure Criteria": "use_eg_for_pressure_criteria",
    "Ice Option": "ice_option",
    "Weir Skew": "weir_skew",
    "Pier Skew": "pier_skew",
    "BR US Left Bank": "bridge_us_left_bank",
    "BR US Right Bank": "bridge_us_right_bank",
    "BR DS Left Bank": "bridge_ds_left_bank",
    "BR DS Right Bank": "bridge_ds_right_bank",
    "XS US Left Bank": "xs_us_left_bank",
    "XS US Right Bank": "xs_us_right_bank",
    "XS DS Left Bank": "xs_ds_left_bank",
    "XS DS Right Bank": "xs_ds_right_bank",
    "US Ineff Left Sta": "us_ineff_left_station",
    "US Ineff Left Elev": "us_ineff_left_elevation",
    "US Ineff Right Sta": "us_ineff_right_station",
    "US Ineff Right Elev": "us_ineff_right_elevation",
    "DS Ineff Left Sta": "ds_ineff_left_station",
    "DS Ineff Left Elev": "ds_ineff_left_elevation",
    "DS Ineff Right Sta": "ds_ineff_right_station",
    "DS Ineff Right Elev": "ds_ineff_right_elevation",
    "Use Override HW Connectivity": "use_override_hw_connectivity",
    "Use Override TW Connectivity": "use_override_tw_connectivity",
    "Use Override HTabIBCurves": "use_override_htab_ib_curves",
    "SNN ID": "snn_id",
    "Default Centerline": "default_centerline",
}

TABLE_INFO_RENAME_MAP = {
    "Centerline Profile (Index)": "centerline_profile_index",
    "Centerline Profile (Count)": "centerline_profile_count",
    "US BR Weir Profile (Index)": "us_deck_high_index",
    "US BR Weir Profile (Count)": "us_deck_high_count",
    "US BR Lid Profile (Index)": "us_deck_low_index",
    "US BR Lid Profile (Count)": "us_deck_low_count",
    "DS BR Weir Profile (Index)": "ds_deck_high_index",
    "DS BR Weir Profile (Count)": "ds_deck_high_count",
    "DS BR Lid Profile (Index)": "ds_deck_low_index",
    "DS BR Lid Profile (Count)": "ds_deck_low_count",
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

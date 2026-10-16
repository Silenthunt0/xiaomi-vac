"""Pure tests for IjaiVacuumDevice control calls."""
from __future__ import annotations

import pytest

from .helpers import FakeMiotDevice, load_device_module


def _last_calls():
    return FakeMiotDevice.instances[-1].calls


def test_start_stop_and_return_home_use_core_actions(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    assert (device.core.start.siid, device.core.start.aiid) == (2, 1)
    assert (device.core.stop.siid, device.core.stop.aiid) == (2, 2)
    assert (device.core.charge.siid, device.core.charge.aiid) == (3, 1)

    device.start()
    device.stop()
    device.return_home()

    assert _last_calls() == [
        ("action", 2, 1, []),
        ("action", 2, 2, []),
        ("action", 3, 1, []),
    ]


def test_device_passes_an_explicit_empty_mapping(monkeypatch):
    device_mod = load_device_module(monkeypatch)

    device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    assert FakeMiotDevice.instances[-1].mapping == {}


def test_pause_falls_back_to_stop_when_no_pause_action(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    assert device.core.pause is None

    device.pause()

    assert _last_calls() == [("action", device.core.stop.siid, device.core.stop.aiid, [])]


def test_pause_uses_real_pause_action_when_present(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "viomi.vacuum.v12")

    assert device.core.pause is not None

    device.pause()

    assert _last_calls() == [
        ("action", device.core.pause.siid, device.core.pause.aiid, [])
    ]


def test_base_station_controls_use_profile_actions(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "xiaomi.vacuum.ov31gl")

    device.set_auto_mop_dry(True)
    device.set_drying_time("3_hours")
    device.start_drying()
    device.stop_drying()
    device.start_mop_wash()
    device.empty_dust_bin()

    assert _last_calls() == [
        ("set", 2, 34, True),
        ("set", 2, 31, 2),
        ("action", 2, 20, []),
        ("action", 2, 32, []),
        ("action", 2, 19, []),
        ("action", 2, 18, []),
    ]


def test_set_fan_speed_uses_value_table(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    device.set_fan_speed("standard")

    assert _last_calls() == [
        (
            "set",
            device.core.fan_speed.siid,
            device.core.fan_speed.piid,
            device.core.fan_speeds["standard"],
        )
    ]


def test_set_fan_speed_rejects_unknown_label(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    with pytest.raises(KeyError):
        device.set_fan_speed("Turbo Plus")

    assert _last_calls() == []


def test_clean_segments_uses_v17_room_clean_action(monkeypatch):
    """set-room-clean (7/3) takes map room ids as a CSV STRING; the
    start-room-sweep action (2/7) wants Mijia ids and fails with map ids
    (verified on v17 hardware, issue #7)."""
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")

    device.clean_segments([10, 12])

    assert _last_calls() == [("action", 7, 3, ["10,12", 0, 1])]


def test_clean_segments_uses_v3_room_clean_action(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")

    device.clean_segments([10, 12])

    assert _last_calls() == [("action", 7, 3, ["10,12", 0, 1])]


def test_ov31gl_room_clean_uses_named_input(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "xiaomi.vacuum.ov31gl")

    device.clean_segments([32])

    assert _last_calls() == [
        (
            "send",
            "action",
            {
                "did": "call-2-16",
                "siid": 2,
                "aiid": 16,
                "in": [{"piid": 15, "value": "32"}],
            },
        )
    ]


@pytest.mark.parametrize(
    "model", ["ijai.vacuum.v17", "ijai.vacuum.v18", "ijai.vacuum.v19"]
)
def test_clean_zone_sets_piid_keyed_zone_then_starts(monkeypatch, model):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", model)

    device.clean_zone(-1.5, 2.25, -0.5, 3.0)

    assert _last_calls() == [
        ("action", 9, 8, [{"piid": 2, "value": "[-1500,2250,-500,3000,1]"}]),
        ("action", 9, 3, []),
    ]


def test_clean_zone_rejects_unverified_point_zone_profile(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")

    assert device.zone_clean_action() is None
    with pytest.raises(ValueError):
        device.clean_zone(0.0, 0.0, 1.0, 1.0)


_X20_MODELS = [
    "xiaomi.vacuum.c107",
    "xiaomi.vacuum.d101",
    "xiaomi.vacuum.d102ev",
    "xiaomi.vacuum.d102gl",
    "xiaomi.vacuum.d109gl",
    "xiaomi.vacuum.ov21gl",
]


@pytest.mark.parametrize("model", _X20_MODELS)
def test_x20_clean_zone_sends_one_zone_sweep_action(monkeypatch, model):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", model)

    device.clean_zone(0.82, 0.88, 2.66, 2.925)

    assert _last_calls() == [
        (
            "action",
            2,
            37,
            ['[{"blocks_region":[820,2925,820,880,2660,880,2660,2925],"blocks_attr":0}]'],
        )
    ]


@pytest.mark.parametrize(
    ("zone", "region"),
    [
        ((2.66, 2.925, 0.82, 0.88), "[820,2925,820,880,2660,880,2660,2925]"),
        ((2.66, 0.88, 0.82, 2.925), "[820,2925,820,880,2660,880,2660,2925]"),
        ((-1.5, 2.25, -0.5, 3.0), "[-1500,3000,-1500,2250,-500,2250,-500,3000]"),
        ((-0.5, 3.0, -1.5, 2.25), "[-1500,3000,-1500,2250,-500,2250,-500,3000]"),
    ],
)
def test_x20_clean_zone_normalises_corners_to_min_max(monkeypatch, zone, region):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "xiaomi.vacuum.d102gl")

    device.clean_zone(*zone)

    assert _last_calls() == [
        ("action", 2, 37, [f'[{{"blocks_region":{region},"blocks_attr":0}}]'])
    ]


@pytest.mark.parametrize("model", ["xiaomi.vacuum.ov31gl", "dreame.vacuum.p2008"])
def test_clean_zone_rejects_models_without_a_zone_capability(monkeypatch, model):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", model)

    assert device.zone_clean_action() is None
    with pytest.raises(ValueError):
        device.clean_zone(0.0, 0.0, 1.0, 1.0)
    assert _last_calls() == []


def test_request_map_upload_prefers_upload_by_mapid_ii(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")

    device.request_map_upload(7)

    assert _last_calls() == [("action", 10, 14, [7])]


def test_request_map_upload_falls_back_to_upload_by_mapid(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")
    FakeMiotDevice.action_results[(10, 14)] = RuntimeError("-1")

    device.request_map_upload(7)

    assert _last_calls() == [
        ("action", 10, 14, [7]),
        ("action", 10, 2, [7]),
    ]


def test_map_upload_actions_lists_upload_by_mapid_ii_then_upload_by_mapid(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")

    assert [(a.siid, a.aiid) for a in device.map_upload_actions()] == [(10, 14), (10, 2)]
    assert _last_calls() == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[3.456,4.555,0.2,1,5.456,4.555,0.233,0]", (5.456, 4.555, 0.233)),
        ("[123,3.456,4.555,0.2,1,-1.5,2.25,1.57,1]", (-1.5, 2.25, 1.57)),
        ('{"deg":0.02,"posArray":[[14,23],[23,22]]}', (23.0, 22.0, 0.0)),
        ("", None),
        ("[]", None),
        ("[1,2,3]", None),
        ("[1,2,3,4,5,6]", None),
        (None, None),
        ("garbage", None),
        # S10 live sample (2026-10-16): leading pose id, 4 per record
        ("[1,0.364375,-0.049650,1.732895,0,0.366826,-0.049477,2.266405,0,0.375094,-0.050460,1.423140,0]",
         (0.375094, -0.050460, 1.423140)),
        # same, but the robot handed over a half-written trailing record
        ("[1,0.364375,-0.049650,1.732895,0,0.366826,-0.049477,2.266405,0,0.375094,-0.05",
         (0.366826, -0.049477, 2.266405)),
        # trailing garbage token
        ("[1,0.364375,-0.049650,1.732895,0,0.366826,-0.049477,2.266405,0,x]",
         (0.366826, -0.049477, 2.266405)),
    ],
)
def test_parse_cleaning_path_position(monkeypatch, raw, expected):
    device_mod = load_device_module(monkeypatch)

    assert device_mod.parse_cleaning_path_position(raw) == expected


def test_status_reads_live_position_only_while_moving(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    path_prop = device.profile.map.current_path
    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = "[7,-1.5,2.25,0.5,1]"

    calls = FakeMiotDevice.instances[-1].calls

    FakeMiotDevice.property_values[(2, 1)] = 5  # sweeping
    moving = device.status()
    assert moving.activity == "cleaning"
    assert moving.position == (-1.5, 2.25, 0.5)
    assert moving.position_source == f"{path_prop.siid}/{path_prop.piid}"
    assert ("get", path_prop.siid, path_prop.piid) in calls

    FakeMiotDevice.property_values[(2, 1)] = 4  # charging
    before = len(calls)
    docked = device.status()
    assert docked.activity == "docked"
    assert docked.position is None
    assert ("get", path_prop.siid, path_prop.piid) not in calls[before:]


def test_status_survives_a_failed_or_empty_path_read(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    path_prop = device.profile.map.current_path
    FakeMiotDevice.property_values[(2, 1)] = 5

    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = RuntimeError("-1")
    assert device.status().position is None

    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = ""
    status = device.status()
    assert status.position is None
    assert status.position_source is None
    assert status.position_raw is None

    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = "weird:1;2;3"
    status = device.status()
    assert status.position is None
    assert status.position_source == f"{path_prop.siid}/{path_prop.piid}:unparsed"
    assert status.position_raw == "str(11):weird:1;2;3"


def test_status_fetches_the_path_window_when_the_property_holds_only_a_pose_id(monkeypatch):
    """S10 shape: 10/5 is '[<pose id>]', the points come from get-cur-path."""
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    path_prop = device.profile.map.current_path
    action = device.profile.map.get_current_path
    FakeMiotDevice.property_values[(2, 1)] = 5
    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = "[1792173122]"
    FakeMiotDevice.action_results[(action.siid, action.aiid)] = {
        "out": [{"piid": 5, "value": "[1792173092,-5.0,1.0,0.1,1,-5.2,1.1,0.2,1]"}]
    }

    status = device.status()

    assert status.position == (-5.2, 1.1, 0.2)
    assert status.position_source == f"{action.siid}/{action.aiid}"
    assert status.position_raw is None
    assert (
        "action", action.siid, action.aiid,
        [{"piid": 15, "value": 1792173122 - 30}, {"piid": 16, "value": 1792173122}],
    ) in _last_calls()


def test_status_widens_the_path_window_when_the_first_reply_is_empty(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    path_prop = device.profile.map.current_path
    action = device.profile.map.get_current_path
    FakeMiotDevice.property_values[(2, 1)] = 5
    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = "[100]"
    replies = iter([{"out": [{"piid": 5, "value": ""}]},
                    {"out": [{"piid": 5, "value": "[1,-1.0,2.0,0.5,1]"}]}])
    fake = FakeMiotDevice.instances[-1] if FakeMiotDevice.instances else None
    original = FakeMiotDevice.call_action_by

    def _reply(self, siid, aiid, params):
        self.calls.append(("action", siid, aiid, params))
        return next(replies)

    monkeypatch.setattr(FakeMiotDevice, "call_action_by", _reply)
    try:
        status = device.status()
    finally:
        monkeypatch.setattr(FakeMiotDevice, "call_action_by", original)

    assert status.position == (-1.0, 2.0, 0.5)
    actions = [c for c in _last_calls() if c[0] == "action"]
    assert [[p["value"] for p in c[3]] for c in actions] == [[70, 100], [0, 100]]
    assert [[p["piid"] for p in c[3]] for c in actions] == [[15, 16], [15, 16]]


def test_status_reports_an_unparsed_path_reply(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    path_prop = device.profile.map.current_path
    action = device.profile.map.get_current_path
    FakeMiotDevice.property_values[(2, 1)] = 5
    FakeMiotDevice.property_values[(path_prop.siid, path_prop.piid)] = "[100]"
    FakeMiotDevice.action_results[(action.siid, action.aiid)] = {"out": [{"piid": 5, "value": "nope"}]}

    status = device.status()

    assert status.position is None
    assert status.position_source == f"{action.siid}/{action.aiid}:unparsed"
    assert status.position_raw == "70,100 -> str:nope"


def test_map_upload_actions_rejects_a_model_without_map_upload(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "dreame.vacuum.p2008")

    with pytest.raises(ValueError):
        device.map_upload_actions()


def test_map_list_parses_map_list_output(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    action = device.profile.map.get_map_list
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {
            "out": [{"piid": 4, "value": '[{"name": "Home", "id": 1, "cur": 1}]'}]
        }
    }

    assert device.map_list() == [{"name": "Home", "id": 1, "cur": 1}]


def test_map_list_returns_empty_for_bad_json(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    action = device.profile.map.get_map_list
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {"out": [{"piid": 4, "value": "not json"}]}
    }

    assert device.map_list() == []


def test_map_list_returns_empty_for_non_list_map_capability(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "dreame.vacuum.p2008")

    assert device.map_list() == []
    assert _last_calls() == []


def test_map_list_reads_viomi_out_piid_11(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "viomi.vacuum.v12")
    action = device.profile.map.get_map_list
    assert action.out_piids == (11,)
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {
            "out": [{"piid": 11, "value": '[{"name": "Home", "id": 1, "cur": true}]'}]
        }
    }

    assert device.map_list() == [{"name": "Home", "id": 1, "cur": True}]


def test_map_list_rejects_viomi_v15_array_of_arrays(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "viomi.vacuum.v15")
    action = device.profile.map.get_map_list
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {
            "out": [{"piid": 11, "value": '[["bkmap", "record", 1620954322, "Map 1", 1]]'}]
        }
    }

    assert device.map_list() == []


def test_map_list_rejects_non_dict_items(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    action = device.profile.map.get_map_list
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {"out": [{"piid": 4, "value": '["Home", "Office"]'}]}
    }

    assert device.map_list() == []


def test_map_list_unwraps_viomi_v22_dict_and_flags_curmapid(monkeypatch):
    """A {"curmapid","map_list","uid"} wrapper is flattened with "cur" derived from curmapid."""
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "viomi.vacuum.v22")
    action = device.profile.map.get_map_list
    out_piid = action.out_piids[0]
    FakeMiotDevice.action_results = {
        (action.siid, action.aiid): {
            "out": [{
                "piid": out_piid,
                "value": (
                    '{"curmapid": 2, "uid": 12345, "map_list": '
                    '[{"name": "Downstairs", "id": 1}, {"name": "Upstairs", "id": 2}, '
                    '{"name": "broken"}, 7]}'
                ),
            }]
        }
    }

    assert device.map_list() == [
        {"name": "Downstairs", "id": 1, "cur": False},
        {"name": "Upstairs", "id": 2, "cur": True},
    ]


def test_status_batches_reads_at_profile_max_properties_on_legacy_v3(monkeypatch):
    """v3 (IJAI_CORE_LEGACY) rejects large batches; profile caps the chunk at 5."""
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v3")
    status_prop = device.core.status
    FakeMiotDevice.property_values = {(status_prop.siid, status_prop.piid): 5}

    device.status()

    assert device.profile.max_properties == 5
    assert FakeMiotDevice.instances[-1].batch_max_properties == [5]


def test_status_sends_unbatched_read_on_v17(monkeypatch):
    """v17 has no max_properties cap — the whole poll goes in one call."""
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "ijai.vacuum.v17")
    status_prop = device.core.status
    FakeMiotDevice.property_values = {(status_prop.siid, status_prop.piid): 5}

    device.status()

    assert device.profile.max_properties is None
    assert FakeMiotDevice.instances[-1].batch_max_properties == [None]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (0, "idle"),
        (1, "drying"),
        (2, "washing_mops"),
        (3, "dust_collection"),
        (99, "unknown"),
    ],
)
def test_base_station_status_maps_working_mode(monkeypatch, mode, expected):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "xiaomi.vacuum.ov31gl")
    FakeMiotDevice.property_values = {
        (2, 2): 14,
        (2, 18): f'{{"mode": {mode}}}',
    }

    assert device.status().base_station_status == expected


def test_base_station_status_reads_tanks_and_drying(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "xiaomi.vacuum.ov31gl")
    FakeMiotDevice.property_values = {
        (2, 2): 14,
        (2, 31): 2,
        (2, 34): True,
        (2, 88): 40,
        (2, 90): 75,
        (2, 97): 1,
        (2, 98): 0,
    }

    status = device.status()

    assert status.drying_time == 2
    assert status.auto_mop_dry is True
    assert status.drying_progress == 40
    assert status.dry_left_time == 75
    assert status.sewage_tank_status == 1
    assert status.water_tank_status == 0
    assert FakeMiotDevice.instances[-1].batch_max_properties == [5]


def test_unsupported_property_and_action_raise_value_error(monkeypatch):
    device_mod = load_device_module(monkeypatch)
    device = device_mod.IjaiVacuumDevice("host", "token", "dreame.vacuum.p2008")

    with pytest.raises(ValueError):
        device.set_alarm(True)

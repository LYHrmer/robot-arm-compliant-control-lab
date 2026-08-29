from pathlib import Path

from compliant_control_lab.experiments import _write_csv as write_planar_csv
from compliant_control_lab.franka_experiments import _write_csv as write_franka_csv
from compliant_control_lab.franka_learning import _write_csv as write_learning_csv
from compliant_control_lab.franka_stress import _write_csv as write_stress_csv


def test_csv_exports_use_repository_safe_lf_endings(tmp_path: Path) -> None:
    writers = (write_planar_csv, write_franka_csv, write_stress_csv, write_learning_csv)
    for index, write_csv in enumerate(writers):
        output_path = tmp_path / f"metrics_{index}.csv"
        write_csv([{"metric": 1.0, "status": "ok"}], output_path)

        payload = output_path.read_bytes()
        assert payload == b"metric,status\n1.0,ok\n"
        assert b"\r" not in payload

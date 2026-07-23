import importlib
from pathlib import Path

from train import convert_xml_annotations


def test_train_module_imports_without_loading_ultralytics_dependency() -> None:
    module = importlib.import_module("train")
    assert module.prepare_dataset is not None


def test_convert_xml_annotations_creates_label_file(tmp_path: Path) -> None:
    source_xml = tmp_path / "sample.xml"
    source_xml.write_text(
        """<annotation>\n  <filename>sample.jpg</filename>\n  <size><width>100</width><height>50</height></size>\n  <object>\n    <name>With Helmet</name>\n    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>\n  </object>\n</annotation>""",
        encoding="utf-8",
    )

    output_path = tmp_path / "labels"
    converted = convert_xml_annotations([source_xml], output_dir=output_path)

    assert converted == [output_path / "sample.txt"]
    assert output_path.joinpath("sample.txt").exists()
    assert output_path.joinpath("sample.txt").read_text(encoding="utf-8").strip() == "0 0.15 0.15 0.1 0.2"

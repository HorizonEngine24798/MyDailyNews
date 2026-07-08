from __future__ import annotations

import unittest

from mydailynews.gui.server import STATIC_DIR, _content_type, _static_path


class GuiServerTests(unittest.TestCase):
    def test_static_path_accepts_nested_assets(self) -> None:
        self.assertEqual(_static_path("/static/js/main.js"), STATIC_DIR / "js" / "main.js")
        self.assertEqual(
            _static_path("/static/assets/MyDailyNewsLogo.png"),
            STATIC_DIR / "assets" / "MyDailyNewsLogo.png",
        )

    def test_png_content_type(self) -> None:
        self.assertEqual(_content_type(STATIC_DIR / "assets" / "MyDailyNewsLogo.png"), "image/png")

    def test_static_path_rejects_traversal(self) -> None:
        unsafe_paths = [
            "/static/",
            "/static/../server.py",
            "/static/js/..%2Fserver.py",
            "/static/js\\main.js",
        ]

        for path in unsafe_paths:
            with self.subTest(path=path):
                with self.assertRaises(FileNotFoundError):
                    _static_path(path)


if __name__ == "__main__":
    unittest.main()

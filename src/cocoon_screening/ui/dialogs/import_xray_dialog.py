from dataclasses import dataclass
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from src.cocoon_screening.paths import default_raw_dir, default_images_dir
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ImportXrayRequest:
    source_directory: Path
    output_directory: Path
    batch_id: str

class ImportXrayDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Import X-rays")
        self.resize(500, 300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Source directory:"))

        source_layout = QHBoxLayout()

        self.source_input = QLineEdit()
        source_layout.addWidget(self.source_input)

        source_button = QPushButton("Browse...")
        source_button.clicked.connect(self.browse_source)
        source_layout.addWidget(source_button)

        layout.addLayout(source_layout)

        # Output
        layout.addWidget(QLabel("Output directory"))

        output_layout = QHBoxLayout()

        self.output_input = QLineEdit()
        output_layout.addWidget(self.output_input)

        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_button)

        layout.addLayout(output_layout)

        # Batch ID
        layout.addWidget(QLabel("Batch ID:"))
        self.batch_id_input = QLineEdit()
        self.batch_id_input.setPlaceholderText("Enter batch ID (e.g., batch_0001)")
        self.batch_id_input.returnPressed.connect(self.start_import)
        layout.addWidget(self.batch_id_input)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        import_button = QPushButton("Import")
        import_button.setDefault(True)
        import_button.setAutoDefault(True)
        import_button.clicked.connect(self.start_import)
        button_layout.addWidget(import_button)

        layout.addLayout(button_layout)

    def browse_source(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select X-ray directory",  str(default_raw_dir())
        )

        if directory:
            self.source_input.setText(directory)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",  str(default_images_dir())
        )

        if directory:
            self.output_input.setText(directory)

    def start_import(self):
        if not self.source_input.text().strip() or not self.output_input.text().strip() or not self.batch_id_input.text().strip():
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            logger.warning("Import failed: Missing input fields.")
            return
        self.accept()  # Close the dialog and indicate success

    def get_import_request(self) -> ImportXrayRequest:
        return ImportXrayRequest(
            source_directory=Path(self.source_input.text()),
            output_directory=Path(self.output_input.text()),
            batch_id=self.batch_id_input.text(),
        )
from datetime import datetime
from functools import partial
from copy import deepcopy
from src.cocoon_screening.ui.worker import Worker
from src.cocoon_screening.processing.annotation_jobs import (
    prepare_annotation_data, prepare_directory, run_annotation_job,
)
from pathlib import Path
from PySide6.QtWidgets import QDialog,QStackedWidget, QButtonGroup,QRadioButton, QGridLayout, QFileDialog, QHBoxLayout, QMainWindow, QLabel, QVBoxLayout, QWidget,QMessageBox, QPushButton, QLineEdit, QComboBox, QCheckBox
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
import cv2
import logging
import numpy as np
import pandas as pd
from enum import Enum
from src.cocoon_screening.domain.review_item import ReviewItem, parse_review_filename
from src.cocoon_screening.processing.stardist import StardistPredictor
from src.cocoon_screening.io.data import get_batch_id, load_images_from_directory, load_labels_from_directory
from src.cocoon_screening.paths import (
    app_version, remembered_directory, remember_directory,
    bundled_configs_dir, default_images_dir, default_review_dir,
    browse_start_directory, normalize_directory,
)
from src.cocoon_screening.ui.utils import array_to_pixmap
from src.cocoon_screening.processing.image_processing import draw_instance_overlay, render_completed_image_overlay, crop_instances
from src.cocoon_screening.domain.label_set import load_labels
from src.cocoon_screening.ui.dialogs.import_xray_dialog import ImportXrayDialog
from src.cocoon_screening.io.batch_importer import import_batch
from src.cocoon_screening.io.atomic import commit_files, recover_commit, image_bytes, csv_bytes, write_bytes
from src.cocoon_screening.domain.classifications import classification_progress, validate_classifications

EXCLUDE_LABEL = "Exclude"
UNSURE_LABEL = "Other (Unsure)"
REVIEW_KEY_COLUMNS = [
    "image_id", "instance_id", "x1", "y1", "x2", "y2",
]

REVIEW_COLUMNS = [
    *REVIEW_KEY_COLUMNS,
    "crop_path",
    "original_class_index",
    "class_index",
    "reviewed_at",
    "overlay_pending",
]
logger = logging.getLogger(__name__)

class AppMode(Enum):
    ANNOTATION = "annotation"
    REVIEW = "review"

class MainWindow(QMainWindow):
    job_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self._busy = False
        self._thread_stopped = False
        self._closing = False
        self._job_callback = None
        self._shortcut_states = []
        self.show_overlay = True
        self.show_review_overlay = False
        self._review_label_maps = {}
        self.init_ui()
        self._worker_thread = QThread(self)
        self._worker = Worker()
        self._worker.moveToThread(self._worker_thread)
        self.job_requested.connect(self._worker.run)
        self._worker.finished.connect(self._on_job_finished)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._on_thread_stopped)
        self._worker_thread.start()

    def init_ui(self):
        self.setWindowTitle(f"Cocoon Annotation Tool {app_version()}")

        ## Main layout
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)

        ## Variables
        self.mode = AppMode.ANNOTATION
        self.stardist_predictor = StardistPredictor()
        labels_path = bundled_configs_dir() / "labels.yaml"
        self.label_set = load_labels(labels_path)
        self.images_path_dict = {}
        self.labels_path_dict = {}
        self.current_crops = {}
        self.current_classifications = pd.DataFrame(columns=["instance_id", "class_index"])
        self.current_cocoon_id = None
        self.current_image = None
        self.loaded_annotation_directory = None
        self.review_items = []
        self.current_review_index = -1
        self.current_review_item = None
        self.loaded_review_directory = None
        self.review_table = pd.DataFrame(columns=REVIEW_COLUMNS)
        self.review_row_lookup = {}
        self.completed_review_keys = set()
        self.review_image_cache = None

        ## Mode selection
        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)

        self.annotation_mode_button = QRadioButton("Annotation")
        self.annotation_mode_button.setStyleSheet("font-weight: bold;")
        self.annotation_mode_button.setFixedWidth(100)
        self.annotation_mode_button.setChecked(True)
        mode_layout.addWidget(self.annotation_mode_button)
        self.annotation_mode_button.toggled.connect(self.switch_to_annotation_mode)

        self.review_mode_button = QRadioButton("Review")
        self.review_mode_button.setStyleSheet("font-weight: bold;")
        self.review_mode_button.setFixedWidth(70)
        mode_layout.addWidget(self.review_mode_button)
        self.review_mode_button.toggled.connect(self.switch_to_review_mode)

        main_layout.addWidget(mode_container, alignment=Qt.AlignCenter)

        ## Directory input
        self.directory_stack = QStackedWidget()

        # Annotation mode directory input
        self.annotation_input_page = QWidget()
        annotation_input_layout = QVBoxLayout(self.annotation_input_page)

        annotation_directory_row = QWidget()
        annotation_directory_layout = QHBoxLayout(annotation_directory_row)

        self.annotation_directory_label = QLabel("Image Directory:")
        annotation_directory_layout.addWidget(self.annotation_directory_label)

        self.annotation_directory_input = QLineEdit()
        self.annotation_directory_input.setPlaceholderText("Enter image directory path...")
        self.annotation_directory_input.returnPressed.connect(
            lambda: self.load_directory(self.annotation_directory_input.text())
            )
        annotation_directory_layout.addWidget(self.annotation_directory_input)

        self.annotation_directory_button = QPushButton("Browse...")
        self.annotation_directory_button.clicked.connect(self.browse_directory)
        annotation_directory_layout.addWidget(self.annotation_directory_button)

        annotation_input_layout.addWidget(annotation_directory_row)

        self.or_label = QLabel("or")
        annotation_input_layout.addWidget(self.or_label, alignment=Qt.AlignCenter)

        self.import_button = QPushButton("Import X-rays")
        self.import_button.clicked.connect(self.open_import_xray_dialog)
        annotation_input_layout.addWidget(self.import_button, alignment=Qt.AlignCenter)

        self.directory_stack.addWidget(self.annotation_input_page)

        # Review mode directory input
        self.review_input_page = QWidget()
        review_directory_layout = QHBoxLayout(self.review_input_page)
        self.review_directory_label = QLabel("Review Directory:")
        review_directory_layout.addWidget(self.review_directory_label)

        self.review_directory_input = QLineEdit()
        self.review_directory_input.setPlaceholderText("Enter review directory path...")
        self.review_directory_input.returnPressed.connect(lambda: self.load_review_directory(self.review_directory_input.text()))
        review_directory_layout.addWidget(self.review_directory_input)

        self.review_directory_button = QPushButton("Browse...")
        self.review_directory_button.clicked.connect(self.browse_directory)
        review_directory_layout.addWidget(self.review_directory_button)

        self.directory_stack.addWidget(self.review_input_page)

        main_layout.addWidget(self.directory_stack)

        ## Annotation Mode Status bar
        self.status_stack = QStackedWidget()
        self.annotation_status_page = QWidget()
        self.annotation_status_page.setObjectName("annotation_status_page")
        self.annotation_status_page.setStyleSheet(
            """
            QWidget#annotation_status_page {
                background-color: #f7f7f7;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            """
        )
        annotation_status_layout = QHBoxLayout(self.annotation_status_page)
        annotation_status_layout.setAlignment(Qt.AlignLeft)

        # batch ID label
        self.batch_id_label = QLabel("Batch ID: ")
        annotation_status_layout.addWidget(self.batch_id_label)

        # Image ID label and dropdown
        image_count_container = QWidget()
        image_count_layout = QHBoxLayout(image_count_container)
        self.image_count_label = QLabel("Image ID: ")
        image_count_layout.addWidget(self.image_count_label)

        self.image_id_label = QLabel(" ")
        image_count_layout.addWidget(self.image_id_label)

        self.image_counter_dropdown = QComboBox()
        self.image_counter_dropdown.setEditable(False)
        self.image_counter_dropdown.setStyleSheet("combobox-popup: 0;")
        # self.image_counter_dropdown.setCurrentText("1")  # Default value
        self.image_counter_dropdown.currentTextChanged.connect(self.update_image_id_from_dropdown)
        image_count_layout.addWidget(self.image_counter_dropdown)

        # Image total count label
        self.image_total_count_label = QLabel("/  ")
        image_count_layout.addWidget(self.image_total_count_label)
        annotation_status_layout.addWidget(image_count_container)

        # Cocoon counter label
        cocoon_count_container = QWidget()
        cocoon_count_layout = QHBoxLayout(cocoon_count_container)
        self.cocoon_count_label = QLabel("Cocoon: ")
        cocoon_count_layout.addWidget(self.cocoon_count_label)

        self.cocoon_idx_dropdown = QComboBox()
        self.cocoon_idx_dropdown.setEditable(False)
        self.cocoon_idx_dropdown.setStyleSheet("combobox-popup: 0;")
        # self.cocoon_idx_dropdown.setCurrentText("1")  # Default value
        self.cocoon_idx_dropdown.currentTextChanged.connect(self.update_cocoon_from_dropdown)
        cocoon_count_layout.addWidget(self.cocoon_idx_dropdown)

        # Cocoon total count label
        self.cocoon_total_count_label = QLabel("/  ")
        cocoon_count_layout.addWidget(self.cocoon_total_count_label)
        annotation_status_layout.addWidget(cocoon_count_container)

        # Image Status
        self.image_completion_label = QLabel("Image status: ")
        annotation_status_layout.addWidget(self.image_completion_label)

        self.status_stack.addWidget(self.annotation_status_page)

        ## Review Mode Status Bar
        self.review_status_page = QWidget()
        self.review_status_page.setObjectName("review_status_page")
        self.review_status_page.setStyleSheet(
            """
            QWidget#review_status_page {
                background-color: #f7f7f7;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            """
        )
        review_status_layout = QHBoxLayout(self.review_status_page)
        review_status_layout.setAlignment(Qt.AlignLeft)

        self.review_item_label = QLabel("Review item: ")
        review_status_layout.addWidget(self.review_item_label)

        self.review_item_counter_dropdown = QComboBox()
        self.review_item_counter_dropdown.setEditable(False)
        self.review_item_counter_dropdown.setStyleSheet("combobox-popup: 0;")
        self.review_item_counter_dropdown.currentIndexChanged.connect(
            self.update_review_item_from_dropdown
            )
        review_status_layout.addWidget(self.review_item_counter_dropdown)

        self.review_item_total_count_label = QLabel("/  ")
        review_status_layout.addWidget(self.review_item_total_count_label)

        self.review_image_id_label = QLabel("Image ID: ")
        review_status_layout.addWidget(self.review_image_id_label)

        self.review_cocoon_instance_label = QLabel("Cocoon: ")
        review_status_layout.addWidget(self.review_cocoon_instance_label)

        self.review_completion_label = QLabel("Review status: ")
        review_status_layout.addWidget(self.review_completion_label)

        self.status_stack.addWidget(self.review_status_page)


        main_layout.addWidget(self.status_stack, alignment=Qt.AlignCenter)

        # Annotation panel
        annotation_container = QWidget()
        annotation_layout = QHBoxLayout(annotation_container)
        annotation_layout.setAlignment(Qt.AlignCenter)
        self.image_display = QLabel()
        self.pixmap = QPixmap()
        self.image_display.setPixmap(self.pixmap)
        annotation_layout.addWidget(self.image_display, alignment=Qt.AlignCenter)

        labels_container = QWidget()
        self.labels_group = QButtonGroup(self)
        self.labels_group.setExclusive(True)
        self.labels_layout = QGridLayout()
        self.labels_layout.setAlignment(Qt.AlignCenter)
        for index, label in enumerate(self.label_set.names):
            shortcut_number = index
            shortcut_number_label = QLabel(str(shortcut_number))
            shortcut_number_label.setFixedWidth(20)
            shortcut_number_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            checkbox = QCheckBox(label)
            checkbox.setStyleSheet(f"QCheckBox {{ color: {self.label_set.color_for(label)}; font-weight: bold; }}")
            checkbox.clicked.connect(self.on_label_selected)

            self.labels_layout.addWidget(shortcut_number_label, index,0)
            self.labels_layout.addWidget(checkbox, index, 1)
            self.labels_group.addButton(checkbox)

        labels_container.setLayout(self.labels_layout)

        annotation_layout.addWidget(labels_container, alignment=Qt.AlignCenter)
        main_layout.addWidget(annotation_container, alignment=Qt.AlignCenter)

        main_layout.addStretch()

        self.shortcut_stack = QStackedWidget()

        self.annotation_shortcut_container = self.create_annotation_shortcut_bar()
        self.shortcut_stack.addWidget(self.annotation_shortcut_container)

        self.review_shortcut_container = self.create_review_shortcut_bar()
        self.shortcut_stack.addWidget(self.review_shortcut_container)

        main_layout.addWidget(self.shortcut_stack)

        self.create_shortcuts()
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure the main window can receive keyboard events

    def switch_to_annotation_mode(self, checked):
        if self._busy:
            for button in (self.annotation_mode_button, self.review_mode_button):
                button.blockSignals(True)
            self.annotation_mode_button.setChecked(self.mode == AppMode.ANNOTATION)
            self.review_mode_button.setChecked(self.mode == AppMode.REVIEW)
            for button in (self.annotation_mode_button, self.review_mode_button):
                button.blockSignals(False)
            return

        if not checked:
            return
        review_dir = self.loaded_review_directory
        self.mode = AppMode.ANNOTATION

        self.directory_stack.setCurrentWidget(self.annotation_input_page)
        self.status_stack.setCurrentWidget(self.annotation_status_page)
        self.shortcut_stack.setCurrentWidget(self.annotation_shortcut_container)

        self.clear_state()

        if review_dir is not None:
            images_dir = review_dir.parent / "images"
            if images_dir.exists():
                if self.confirm_relative_directory(
                    "Open Annotation Directory",
                    f"Open the image directory belonging to this review directory?\n\n{images_dir}",
                ):
                    self.annotation_directory_input.setText(str(images_dir))
                    self.load_directory(str(images_dir))


    def switch_to_review_mode(self, checked):
        if self._busy:
            for button in (self.annotation_mode_button, self.review_mode_button):
                button.blockSignals(True)
            self.annotation_mode_button.setChecked(self.mode == AppMode.ANNOTATION)
            self.review_mode_button.setChecked(self.mode == AppMode.REVIEW)
            for button in (self.annotation_mode_button, self.review_mode_button):
                button.blockSignals(False)
            return
        """
        - turn on review mode
        - change image directory label to "Review Directory"
        -
        - hide or label/ import xray button
        - change the directory input to default "for_review" of last image directory or set to placeholder if no previous directory
        - hide shortcuts for image navigation
        - remove last displayed cocoon, remove image id, cocoon id
        - replace image status with review status
        - initiate review object / load review directory with crops and metadata -> list of reviewitems
        - set display to first review cocoon, set image id, set cocoon id, set review status
        """
        if not checked:
            return
        image_dir = self.loaded_annotation_directory
        try:
            self.save_before_annotation_navigation()
        except Exception as error:
            self.annotation_mode_button.blockSignals(True)
            self.review_mode_button.blockSignals(True)
            try:
                self.annotation_mode_button.setChecked(True)
            finally:
                self.annotation_mode_button.blockSignals(False)
                self.review_mode_button.blockSignals(False)
            self.show_error("Save Error", "Could not save before switching mode.", error)
            return
        self.mode = AppMode.REVIEW

        self.directory_stack.setCurrentWidget(self.review_input_page)
        self.status_stack.setCurrentWidget(self.review_status_page)
        self.shortcut_stack.setCurrentWidget(self.review_shortcut_container)

        self.clear_state()

        self.pixmap = QPixmap()
        self.image_display.setPixmap(self.pixmap)

        if image_dir is not None:
            default_review_dir = image_dir.parent / "for_review"
            if default_review_dir.exists():
                if self.confirm_relative_directory(
                    "Open Review Directory",
                    f"Open the review directory belonging to this image directory?\n\n{default_review_dir}",
                ):
                    self.review_directory_input.setText(str(default_review_dir))
                    self.load_review_directory(str(default_review_dir))




    def set_default_review_directory(self):

        pass

    def confirm_relative_directory(self, title, message):
        return (
            QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            == QMessageBox.Yes
        )

    def open_import_xray_dialog(self):
        dialog = ImportXrayDialog(self)
        if dialog.exec() != QDialog.Accepted:
            logger.info("Import cancelled by user.")
            return

        request = dialog.get_import_request()
        self.import_xray_batch(request)

    def import_xray_batch(self, request):
        try:
            result = import_batch(
                batch_directory=request.source_directory,
                output_directory=request.output_directory,
                image_registry_path=request.output_directory.parent / "image_registry.csv",
                batch_id=request.batch_id,
            )

            summary = (
                f"Batch {result.batch_id}: {result.discovered} files found.\n"
                f"Imported: {result.imported}\n"
                f"Duplicates skipped: {result.skipped_duplicates}\n"
                f"Failed: {result.failed}"
            )
            if result.failed or not result.discovered:
                QMessageBox.warning(self, "Import Results", summary)
            else:
                QMessageBox.information(self, "Import Complete", summary)

            logger.info(
                "Batch %s successfully imported from %s to %s. From %s discovered files, %s imported, %s duplicates skipped, and %s failed",
                result.batch_id,
                request.source_directory,
                result.output_directory,
                result.discovered,
                result.imported,
                result.skipped_duplicates,
                result.failed,
            )
            if result.imported:
                self.load_directory(str(result.output_directory))

        except Exception as e:
            self.show_error("Import Error", f"Failed to import batch {request.batch_id}:\n{e}", e)

    def display_cocoon(self, text):
        if not str(text).isdigit() or int(text) not in self.current_crops:
            raise ValueError(f"Invalid cocoon ID: {text}")
        instance_id = int(text)
        pixmap = self.annotation_crop_pixmap(self.current_crops[instance_id])
        self.current_cocoon_id = instance_id
        self.set_review_selection()
        if not self.restore_saved_classification_selection():
            self.labels_group.buttons()[0].setChecked(True)
        self.image_display.setPixmap(pixmap)
        self.cocoon_count_label.setText(f"Cocoon {instance_id}: ")


    def selected_classification(self):
        selected_button = self.labels_group.checkedButton()
        if selected_button:
            return selected_button.text()
        return None

    def classification_file_path(self):
        return self.current_image.parent.parent / "labels" / f"{self.current_image.stem}.csv"

    def create_shortcuts(self):
        # Next cocoon
        next_cocoon_right_shortcut = QShortcut(QKeySequence("Right"), self)
        next_cocoon_right_shortcut.activated.connect(self.next_cocoon)

        # Previous cocoon
        prev_cocoon_shortcut = QShortcut(QKeySequence("Left"), self)
        prev_cocoon_shortcut.activated.connect(self.prev_cocoon)

        # Next image
        next_image_shortcut = QShortcut(QKeySequence("Ctrl+Right"), self)
        next_image_shortcut.activated.connect(self.next_image)

        # Previous image
        prev_image_shortcut = QShortcut(QKeySequence("Ctrl+Left"), self)
        prev_image_shortcut.activated.connect(self.prev_image)

        # Overlay on/off
        toggle_overlay_shortcut = QShortcut(QKeySequence("Space"), self)
        toggle_overlay_shortcut.activated.connect(self.toggle_overlay)

        # Label selection shortcuts
        for i in range(min(len(self.label_set), 9)):
            label_shortcut = QShortcut(QKeySequence(f"{i}"), self)
            label_shortcut.activated.connect(lambda idx=i: self.select_label_by_index(idx))

    def select_label_by_index(self, index):
        if getattr(self, "_busy", False):
            return
        if not 0 <= index < len(self.label_set):
            return
        label_name = self.label_set.names[index]
        for button in self.labels_group.buttons():
            if button.text() == label_name:
                button.setChecked(True)
                self.on_label_selected()
                return

    def current_image_is_fully_classified(self):
        return classification_progress(
            self.current_classifications, self.current_crops, len(self.label_set)
        )[1]

    def update_image_completion_status(self):
        if self.current_image_is_fully_classified():
            self.image_completion_label.setText("Image status: Complete")
            self.image_completion_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            labelled_count = classification_progress(
                self.current_classifications, self.current_crops, len(self.label_set)
            )[0]
            total_count = len(self.current_crops)
            self.image_completion_label.setText(
                f"Image status: {labelled_count}/{total_count}"
            )
            self.image_completion_label.setStyleSheet(
                "color: black; font-weight: normal;"
            )

    def next_cocoon(self):
        if getattr(self, "_busy", False):
            return
        if self.mode == AppMode.REVIEW:
            self.move_review_to_next_item(1)
            return
        current_index = self.cocoon_idx_dropdown.currentIndex()
        if current_index < len(self.current_crops) - 1:
            self.cocoon_idx_dropdown.setCurrentIndex(current_index + 1)
        else:
            # If at the last cocoon, move to the next image
            self.next_image()

    def prev_cocoon(self):
        if self._busy:
            return
        if self.mode == AppMode.REVIEW:
            self.move_review_to_next_item(-1)
            return
        current_index = self.cocoon_idx_dropdown.currentIndex()
        if current_index > 0:
            self.cocoon_idx_dropdown.setCurrentIndex(current_index - 1)
        else:
            # If at the first cocoon, move to the previous image
            self.prev_image()

    def next_image(self):
        if getattr(self, "_busy", False):
            return
        if self.mode == AppMode.REVIEW:
            return
        current_index = self.image_counter_dropdown.currentIndex()
        if current_index < len(self.images_path_dict) - 1:
            self.image_counter_dropdown.setCurrentIndex(current_index + 1)
        else:
            QMessageBox.information(self, "Navigation", "There is no next image.")

    def prev_image(self):
        if self._busy:
            return
        if self.mode == AppMode.REVIEW:
            return
        current_index = self.image_counter_dropdown.currentIndex()
        if current_index > 0:
            self.image_counter_dropdown.setCurrentIndex(current_index - 1)
        else:
            QMessageBox.information(self, "Navigation", "There is no previous image.")

    def on_label_selected(self):
        if getattr(self, "_busy", False):
            return
        try:
            self._on_label_selected()
        except Exception as error:
            if self.mode == AppMode.ANNOTATION:
                self.set_review_selection()
                self.restore_saved_classification_selection()
            self.show_error("Error", "The selected class could not be applied.", error)

    def _on_label_selected(self):
        if self.mode == AppMode.REVIEW:
            if self.current_review_item is None:
                logger.warning("No review item selected for classification.")
                return
            if not self.save_current_review_classification():
                logger.warning("Failed to save classification for review item: %s", self.current_review_item)
                return
            if not self.finish_review_if_complete():
                self.move_review_to_next_item(1)
            return

        current_classification = self.selected_classification()

        if current_classification == EXCLUDE_LABEL:
            self.exclude_current_cocoon()
            return

        if current_classification == UNSURE_LABEL:
            self.save_current_cocoon_for_review()

        self.save_current_cocoon_classification()
        self.next_cocoon()

    def for_review_directory(self):
        return self.current_image.parent.parent / "for_review"

    def label_map_file_path(self):
        return self.labels_path_dict[self.current_image.stem]

    def save_current_cocoon_for_review(self):
        if self.current_cocoon_id is None:
            return
        if self.current_cocoon_id not in self.current_crops:
            logger.warning("Cannot save review crop. Invalid cocoon instance ID %s for image %s",
                self.current_cocoon_id,
                self.current_image.stem,
            )
            return

        crop = self.current_crops[self.current_cocoon_id]
        review_directory = self.for_review_directory()
        review_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        image_id = Path(self.current_image).stem
        instance_id = crop["instance_id"]
        x1, y1, x2, y2 = crop["bounds"]
        crop_filename = f"{image_id}_instance_{instance_id}_x{x1}_y{y1}-x{x2}_y{y2}.png"
        crop_path = review_directory / crop_filename
        item = parse_review_filename(crop_path)

        crop_image = crop["image"]
        scaled_crop = cv2.resize(
            crop_image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_LINEAR,
        )

        metadata_path = review_directory / "for_review.csv"

        row = pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_path": str(crop_path),
                    "original_class_index": self.label_set.index_of(UNSURE_LABEL),
                    "class_index": pd.NA,
                    "reviewed_at": pd.NA,
                }
            ]
        )

        review_table = self.read_review_table(review_directory)
        row_index = self.ensure_review_row(review_table, item)

        # A newly queued item must not inherit its old review decision.
        for column in row.columns:
            review_table.at[row_index, column] = row.iloc[0][column]

        self.write_csv_atomic(review_table, metadata_path)
        # Reset completion durably BEFORE replacing an existing crop.
        # A crop-write failure now leaves a pending record, never an old decision.
        self.write_image_atomic(scaled_crop, crop_path)
        logger.info(
            "Cocoon instance %s in image %s not clear -> saved for review under %s",
            instance_id,
            image_id,
            crop_path
        )

    def next_instance_id_after_removal(self, removed_instance_id):
        instance_ids = list(self.current_crops.keys())

        removed_index = instance_ids.index(removed_instance_id)

        remaining_instance_ids = [
            instance_id
            for instance_id in instance_ids
            if instance_id != removed_instance_id
        ]

        if not remaining_instance_ids:
            return None

        if removed_index < len(remaining_instance_ids):
            return remaining_instance_ids[removed_index]
        else: # last element was removed
            return remaining_instance_ids[-1]

    def exclude_current_cocoon(self):
        try:
            self._exclude_current_cocoon()
        except Exception as error:
            self.set_review_selection()
            self.restore_saved_classification_selection()
            self.show_error("Error", "The cocoon could not be removed from the label map.", error)

    def _exclude_current_cocoon(self):
        if self.current_cocoon_id is None:
            return
        instance_id = self.current_cocoon_id
        next_id = self.next_instance_id_after_removal(instance_id)
        journal = self.annotation_journal(self.current_image)
        recover_commit(journal)
        prepared = self.prepare_annotation_image(self.current_image, self.labels_path_dict)
        crops = prepared["crops"].copy()
        classifications = prepared["classifications"].copy()
        label_path = prepared["label_path"]
        label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_image is None:
            raise ValueError(f"Could not read label map: {label_path}")
        if label_image.ndim == 3:
            label_image = label_image[:, :, 0]
        label_image[label_image == instance_id] = 0
        crops.pop(instance_id, None)
        classifications = classifications.loc[
            classifications["instance_id"] != instance_id
        ].copy()
        image = cv2.imread(str(self.current_image), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {self.current_image}")
        overlay = render_completed_image_overlay(
            image, crops, classifications, self.label_set, alpha=0.15
        )
        overlay_path = self.current_image.parent.parent / "overlays" / f"{self.current_image.stem}_class_overlay.png"
        # All outputs are staged before any existing file is changed. On failure
        # reopening this image finishes the journal before exposing its state.
        commit_files(journal, {
            label_path: image_bytes(label_image, label_path),
            self.classification_file_path(): csv_bytes(classifications),
            overlay_path: image_bytes(overlay, overlay_path),
        })
        prepared["crops"] = crops
        prepared["classifications"] = classifications
        self.activate_annotation_image(self.current_image, prepared, next_id)


    def update_image_id_from_dropdown(self, text):
        if not str(text).isdigit():
            return

        index = int(text) - 1  # Convert to 0-based index

        if 0 <= index < len(self.images_path_dict):
            self.switch_to_image(index, save_previous=True)

        self.setFocus()  # Set focus back to the main window to ensure keyboard shortcuts work

    def _restore_image_dropdown(self):
        self.image_counter_dropdown.blockSignals(True)
        try:
            index = (list(self.images_path_dict).index(self.current_image.stem)
                     if self.current_image is not None else -1)
            self.image_counter_dropdown.setCurrentIndex(index)
        finally:
            self.image_counter_dropdown.blockSignals(False)

    def switch_to_image(self, index, save_previous=True):
        if self._busy:
            self._restore_image_dropdown()
            return
        try:
            image_ids = list(self.images_path_dict)
            if not 0 <= index < len(image_ids):
                return
            image_path = self.images_path_dict[image_ids[index]]
            load = partial(prepare_annotation_data, image_path,
                           dict(self.labels_path_dict), self.stardist_predictor,
                           deepcopy(self.label_set))
            self._restore_image_dropdown()
            self._start_annotation_job(
                load, partial(self.activate_annotation_image, image_path),
                "Saving and loading image...", save_previous,
            )
        except Exception as error:
            self._restore_image_dropdown()
            self.show_error("Error", "Failed to open image", error)

    @staticmethod
    def annotation_journal(image_path):
        return image_path.parent.parent / f".{image_path.stem}.exclusion.pending.json"

    def save_before_annotation_navigation(self):
        if self.current_image is not None and not self.annotation_journal(self.current_image).exists():
            self.save_current_cocoon_classification()

    def prepare_annotation_image(self, image_path, label_paths):
        # Synchronous compatibility path for exclusion; pixmaps stay on the GUI thread.
        return prepare_annotation_data(image_path, label_paths,
                                       self.stardist_predictor, self.label_set)

    def annotation_crop_pixmap(self, crop, show_overlay=None):
        if show_overlay is None:
            show_overlay = self.show_overlay
        if show_overlay:
            image = draw_instance_overlay(crop["image"], crop["mask"],
                                          color=(0, 255, 182), alpha=0.35)
        else:
            # Raw cocoon: no mask tint and no contour, so a faint density difference
            # is not hidden by the segmentation outline.
            image = crop["image"]
        pixmap = array_to_pixmap(image)
        return pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def toggle_overlay(self):
        """Space: switch the segmentation overlay on and off for the cocoon on screen."""
        if getattr(self, "_busy", False):
            return
        if getattr(self, "mode", None) == AppMode.REVIEW:
            self.show_review_overlay = not self.show_review_overlay
            self.refresh_review_pixmap()
            return
        self.show_overlay = not self.show_overlay
        self.refresh_cocoon_pixmap()

    def review_instance_mask(self, item, shape):
        """Mask of one review crop, rebuilt from the label map beside the review folder.

        Review crops are saved raw (save_current_cocoon_for_review stores crop["image"]),
        so the overlay has to be reconstructed. Returns None when the label map is not
        reachable - a review folder can be opened on its own - and the crop then stays raw.
        """
        label_path = item.crop_path.parent.parent / "labels" / f"{item.image_id}.png"
        if not label_path.exists():
            return None
        label_map = self._review_label_maps.get(item.image_id)
        if label_map is None:
            label_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if label_map is None:
                return None
            self._review_label_maps[item.image_id] = label_map
        x1, y1, x2, y2 = item.bounds                      # inclusive, as crop_instances writes them
        if y2 >= label_map.shape[0] or x2 >= label_map.shape[1]:
            return None
        mask = label_map[y1:y2 + 1, x1:x2 + 1] == item.instance_id
        if not mask.any():
            return None
        height, width = shape[:2]                         # the saved crop is scaled 2x
        return cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def review_crop_pixmap(self, item, crop):
        if self.show_review_overlay:
            mask = self.review_instance_mask(item, crop.shape)
            if mask is not None:
                crop = draw_instance_overlay(crop, mask, color=(0, 255, 182), alpha=0.35)
            else:
                logger.info("No label map for %s; showing the review crop without overlay",
                            item.image_id)
        return array_to_pixmap(crop)

    def refresh_review_pixmap(self):
        """Redraw the review crop on screen; the saved decision is untouched."""
        item = getattr(self, "current_review_item", None)
        crop = getattr(self, "current_review_crop", None)
        if item is None or crop is None:
            return
        self.image_display.setPixmap(self.review_crop_pixmap(item, crop))

    def refresh_cocoon_pixmap(self):
        """Redraw the current cocoon only; selection and saved labels are untouched."""
        crop = getattr(self, "current_crops", {}).get(getattr(self, "current_cocoon_id", None))
        if crop is None:
            return
        self.image_display.setPixmap(self.annotation_crop_pixmap(crop))

    def activate_annotation_image(self, image_path, prepared, selected_id=None):
        crops = prepared["crops"]
        if selected_id not in crops:
            selected_id = next(iter(crops), None)
        pixmap = None
        if selected_id is not None:
            pixmap = prepared["pixmaps"].get(selected_id) if self.show_overlay else None
            if pixmap is None:
                pixmap = self.annotation_crop_pixmap(crops[selected_id])
        self.current_image = image_path
        self.current_crops = crops
        self.current_classifications = prepared["classifications"]
        self.current_cocoon_id = selected_id
        self.labels_path_dict[image_path.stem] = prepared["label_path"]
        self.image_id_label.setText(image_path.stem)
        self.batch_id_label.setText(f"Batch ID: {prepared['batch_id']}")
        for dropdown in (self.image_counter_dropdown, self.cocoon_idx_dropdown):
            dropdown.blockSignals(True)
        try:
            self.image_counter_dropdown.setCurrentIndex(list(self.images_path_dict).index(image_path.stem))
            self.cocoon_idx_dropdown.clear()
            self.cocoon_idx_dropdown.addItems([str(i) for i in crops])
            self.cocoon_idx_dropdown.setCurrentText(str(selected_id) if selected_id is not None else "")
        finally:
            self.image_counter_dropdown.blockSignals(False)
            self.cocoon_idx_dropdown.blockSignals(False)
        self.cocoon_total_count_label.setText(f"/ {len(crops)}")
        self.set_review_selection()
        if selected_id is None:
            self.image_display.clear()
            self.cocoon_count_label.setText("Cocoon: ")
        else:
            self.image_display.setPixmap(pixmap)
            self.cocoon_count_label.setText(f"Cocoon {selected_id}: ")
            if not self.restore_saved_classification_selection():
                self.labels_group.buttons()[0].setChecked(True)
        self.update_image_completion_status()


    def restore_saved_classification_selection(self):
        """
        Checks if the newly chosen cocoon was previously classified and sets the corresponding label as checked.
        """
        matches = self.current_classifications.loc[self.current_classifications["instance_id"] == self.current_cocoon_id, "class_index"]

        if matches.empty:
            return False  # No previous classification found for this cocoon
        saved_class_index = matches.iloc[0]

        if pd.isna(saved_class_index):
            return False  # No previous classification found for this cocoon
        saved_class_index = int(saved_class_index)
        for button in self.labels_group.buttons():
            if button.text() == self.label_set.name_for(saved_class_index):
                button.setChecked(True)
                return True  # Successfully restored the saved classification
        return False  # No matching label found

    def update_cocoon_from_dropdown(self, text):
        if getattr(self, "_busy", False):
            return
        if not text:
            return
        previous_id = self.current_cocoon_id
        try:
            self.save_current_cocoon_classification()
            self.display_cocoon(text)
            self.setFocus()
        except Exception as error:
            self.current_cocoon_id = previous_id
            self.cocoon_idx_dropdown.blockSignals(True)
            try:
                self.cocoon_idx_dropdown.setCurrentText(str(previous_id))
            finally:
                self.cocoon_idx_dropdown.blockSignals(False)
            self.show_error("Navigation Error", "Could not save or open the cocoon.", error)


    def save_completed_image_overlay(self) -> None:
        if self.current_image is None or not self.current_image_is_fully_classified():
            return
        image_path = self.current_image
        out_path = image_path.parent.parent / "overlays" / f"{image_path.stem}_class_overlay.png"
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            result = render_completed_image_overlay(
                image, self.current_crops, self.current_classifications,
                self.label_set, alpha=0.15,
            )
            self.write_image_atomic(result, out_path)
        except Exception:
            self.image_completion_label.setText("Image status: Overlay save failed - retry")
            raise
        self.update_image_completion_status()


    def save_current_cocoon_classification(self):
        if self.current_cocoon_id is None:
            return
        if self.annotation_journal(self.current_image).exists():
            raise RuntimeError("An exclusion is pending. Reload the image directory to recover it.")
        if self.current_cocoon_id not in self.current_crops:
            logger.warning("Invalid cocoon index %s for image %s", self.current_cocoon_id, self.current_image.stem)
            return
        current_crop = self.current_crops[self.current_cocoon_id]
        instance_id = current_crop["instance_id"]
        selected_class = self.selected_classification()
        if selected_class is None:
            return
        class_index = self.label_set.index_of(selected_class)
        updated = self.current_classifications.copy()
        row_mask = updated["instance_id"] == instance_id
        if row_mask.any():
            updated.loc[row_mask, "class_index"] = class_index
        else:
            updated = pd.concat([updated, pd.DataFrame([{
                "instance_id": instance_id,
                "class_index": class_index,
            }])], ignore_index=True)

        updated = validate_classifications(updated, len(self.label_set))
        self.write_csv_atomic(updated, self.classification_file_path())
        self.current_classifications = updated
        current_crop["class_index"] = class_index
        self.update_image_completion_status()
        self.save_completed_image_overlay()

    def _annotation_save_snapshot(self):
        if (self.mode != AppMode.ANNOTATION or self.current_image is None
                or self.current_cocoon_id not in self.current_crops):
            return None
        selected = self.selected_classification()
        if selected is None:
            return None
        table = self.current_classifications.copy(deep=True)
        instance_id = self.current_crops[self.current_cocoon_id]["instance_id"]
        class_index = self.label_set.index_of(selected)
        mask = table["instance_id"] == instance_id
        if mask.any():
            table.loc[mask, "class_index"] = class_index
        else:
            table = pd.concat([table, pd.DataFrame([{
                "instance_id": instance_id, "class_index": class_index,
            }])], ignore_index=True)
        # Arrays are read-only inputs to this job. Editing is disabled until it finishes.
        crops = {key: dict(crop) for key, crop in self.current_crops.items()}
        return self.current_image, table, crops

    def _set_busy(self, busy, message=""):
        self._busy = busy
        self.centralWidget().setEnabled(not busy)
        if busy:
            self._shortcut_states = [(shortcut, shortcut.isEnabled())
                                     for shortcut in self.findChildren(QShortcut)]
            for shortcut, _ in self._shortcut_states:
                shortcut.setEnabled(False)
        else:
            for shortcut, enabled in self._shortcut_states:
                shortcut.setEnabled(enabled)
            self._shortcut_states = []
        self.statusBar().showMessage(message)

    def _start_annotation_job(self, load, callback, message, save_previous=True):
        if self._busy or self._thread_stopped:
            return
        snapshot = self._annotation_save_snapshot() if save_previous else None
        operation = partial(run_annotation_job, snapshot, deepcopy(self.label_set), load)
        self._job_callback = callback
        self._set_busy(True, message)
        self.job_requested.emit(operation)

    @Slot(object, object)
    def _on_job_finished(self, result, error):
        callback, self._job_callback = self._job_callback, None
        try:
            if result is not None:
                if result.saved is not None:
                    path, table = result.saved
                    if path == self.current_image:
                        self.current_classifications = table
                        for row in table.itertuples():
                            if row.instance_id in self.current_crops:
                                self.current_crops[row.instance_id]["class_index"] = row.class_index
                        self.update_image_completion_status()
                if result.overlay_failed:
                    self.image_completion_label.setText("Image status: Overlay save failed - retry")
                error = result.error or error
            if error is not None:
                raise error
            if callback is not None:
                callback(result.payload)
        except Exception as failure:
            self._restore_image_dropdown()
            message = "Could not complete the operation."
            if result is not None and result.saved is not None:
                message += " The previous annotation was saved."
            self.show_error("Error", message, failure)
        finally:
            # Shutdown keeps the controls disabled until the thread has exited.
            if not self._closing:
                self._set_busy(False)

    def _begin_shutdown(self, unused=None):
        self._closing = True
        self.statusBar().showMessage("Closing...")
        self._worker_thread.quit()

    @Slot()
    def _on_thread_stopped(self):
        self._thread_stopped = True
        self.close()

    def closeEvent(self, event):
        if self._thread_stopped:
            super().closeEvent(event)
            return
        event.ignore()
        if self._busy:
            self.statusBar().showMessage("Please wait for the operation to finish, then close again.")
            return
        try:
            self._start_annotation_job(None, self._begin_shutdown, "Saving before closing...")
        except Exception as error:
            self.show_error("Save Error", "Could not save before closing.", error)

    def browse_directory(self):
        if self._busy:
            return
        if self.mode == AppMode.REVIEW:
            title = "Select Review Directory"
            start = browse_start_directory(
                self.review_directory_input.text(), self.loaded_review_directory,
                remembered_directory("review"), default_review_dir(),
            )
        else:
            title = "Select Image Directory"
            start = browse_start_directory(
                self.annotation_directory_input.text(), self.loaded_annotation_directory,
                remembered_directory("annotation"), default_images_dir(),
            )
        directory = QFileDialog.getExistingDirectory(self, title, str(start))
        if directory and self.mode == AppMode.ANNOTATION:
            remember_directory("annotation", directory)
            self.load_directory(directory)
        elif directory and self.mode == AppMode.REVIEW:
            remember_directory("review", directory)
            self.load_review_directory(directory)

    def load_directory(self, directory):
        if self._busy:
            return
        try:
            directory = normalize_directory(directory)
            load = partial(prepare_directory, directory, self.stardist_predictor,
                           deepcopy(self.label_set))
            self._start_annotation_job(load, self._activate_directory, "Loading directory...")
        except Exception as error:
            self.show_error("Error", "Failed to load directory", error)

    def clear_state(self):
        self.images_path_dict = {}
        self.labels_path_dict = {}
        self.current_crops = {}
        self.current_classifications = pd.DataFrame(
            columns=["instance_id", "class_index"]
        )
        self.current_cocoon_id = None
        self.current_image = None
        self.loaded_annotation_directory = None

        self.image_counter_dropdown.blockSignals(True)
        self.image_counter_dropdown.clear()
        self.image_counter_dropdown.blockSignals(False)

        self.cocoon_idx_dropdown.blockSignals(True)
        self.cocoon_idx_dropdown.clear()
        self.cocoon_idx_dropdown.blockSignals(False)

        self.batch_id_label.setText("Batch ID: ")
        self.image_id_label.setText(" ")
        self.image_total_count_label.setText("/  ")
        self.cocoon_count_label.setText("Cocoon: ")
        self.cocoon_total_count_label.setText("/  ")

        was_exclusive = self.labels_group.exclusive()
        self.labels_group.setExclusive(False)

        for button in self.labels_group.buttons():
            button.setChecked(False)

        self.labels_group.setExclusive(was_exclusive)

        self.image_completion_label.setText("Image status: ")
        self.image_completion_label.setStyleSheet("color: black; font-weight: normal;")

        self.review_items = []
        self.current_review_index = -1
        self.current_review_item = None
        self.loaded_review_directory = None
        self.review_table = pd.DataFrame(columns=REVIEW_COLUMNS)
        self.review_row_lookup = {}
        self.completed_review_keys = set()
        self.review_image_cache = None

        self.review_item_total_count_label.setText("/  ")

        self.review_item_counter_dropdown.blockSignals(True)
        self.review_item_counter_dropdown.clear()
        self.review_item_counter_dropdown.blockSignals(False)

        self.review_image_id_label.setText("Image ID: ")
        self.review_cocoon_instance_label.setText("Cocoon: ")

        self.review_completion_label.setText("Review status: ")
        self.review_completion_label.setStyleSheet("color: black; font-weight: normal;")

        self.image_display.clear()

    def _activate_directory(self, payload):
        directory, images, labels, image_path, prepared = payload
        # Validate the initial pixmap before replacing the loaded dataset.
        if prepared is not None and prepared["crops"]:
            first = next(iter(prepared["crops"]))
            prepared["pixmaps"][first] = self.annotation_crop_pixmap(prepared["crops"][first])
        self.clear_state()
        self.loaded_annotation_directory = directory
        self.annotation_directory_input.setText(str(directory))
        self.images_path_dict = images
        self.labels_path_dict = labels
        self.image_counter_dropdown.blockSignals(True)
        try:
            self.image_counter_dropdown.clear()
            self.image_counter_dropdown.addItems([str(i + 1) for i in range(len(images))])
        finally:
            self.image_counter_dropdown.blockSignals(False)
        self.image_total_count_label.setText(f"/ {len(images)}")
        if prepared is not None:
            self.activate_annotation_image(image_path, prepared)


    def load_review_directory(self, directory):
        if getattr(self, "_busy", False):
            return
        try:
            self._load_review_directory(directory)
        except Exception as error:
            self.show_error("Error", "Failed to load review directory", error)

    @staticmethod
    def write_csv_atomic(table, path):
        write_bytes(path, csv_bytes(table))

    @staticmethod
    def write_image_atomic(image, path):
        write_bytes(path, image_bytes(image, path))

    def read_review_table(self, directory):
        path = Path(directory) / "for_review.csv"

        if not path.exists():
            return pd.DataFrame(columns=REVIEW_COLUMNS)

        table = pd.read_csv(path, dtype={"image_id": "string"})

        required = set(REVIEW_KEY_COLUMNS + ["class_index"])
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"Review CSV is missing columns: {sorted(missing)}"
            )

        # Preserve the old value for legacy records. Without a completion
        # timestamp, these records still require explicit review.
        if "original_class_index" not in table.columns:
            table["original_class_index"] = table["class_index"]

        for column in REVIEW_COLUMNS:
            if column not in table.columns:
                table[column] = pd.NA

        for column in REVIEW_KEY_COLUMNS[1:]:
            table[column] = pd.to_numeric(
                table[column], errors="raise"
            ).astype("Int64")

            if table[column].isna().any():
                raise ValueError(f"Missing review identity: {column}")

        table["class_index"] = pd.to_numeric(
            table["class_index"], errors="coerce"
        )
        table["original_class_index"] = pd.to_numeric(
            table["original_class_index"], errors="coerce"
        )

        # Permit assigning ISO timestamp strings to an initially empty column.
        table["reviewed_at"] = table["reviewed_at"].astype("object")
        table["crop_path"] = table["crop_path"].astype("object")
        table["overlay_pending"] = pd.to_numeric(
            table["overlay_pending"], errors="raise"
        ).fillna(0).astype("Int64")
        if not table["overlay_pending"].isin([0, 1]).all():
            raise ValueError("Invalid overlay_pending value in review CSV.")

        return table


    @staticmethod
    def review_key(item):
        return (item.image_id, item.instance_id, *item.bounds)


    def review_row_index(self, table, item):
        mask = pd.Series(True, index=table.index, dtype=bool)

        for column, value in zip(
            REVIEW_KEY_COLUMNS, self.review_key(item)
        ):
            mask &= table[column].eq(value).fillna(False)

        matches = table.index[mask]

        if len(matches) > 1:
            raise ValueError(
                f"Duplicate CSV records for {item.crop_path.name}"
            )

        return matches[0] if len(matches) else None


    def ensure_review_row(self, table, item):
        index = self.review_row_index(table, item)

        if index is None:
            index = len(table)
            record = dict(zip(REVIEW_KEY_COLUMNS, self.review_key(item)))
            record.update({
                "crop_path": str(item.crop_path),
                "original_class_index": self.label_set.index_of(UNSURE_LABEL),
                "class_index": pd.NA,
                "reviewed_at": pd.NA,
            })

            for column, value in record.items():
                table.at[index, column] = value
        else:
            # Paths can change when the dataset directory is moved.
            table.at[index, "crop_path"] = str(item.crop_path)

        return index


    def review_row_is_complete(self, row):
        value = pd.to_numeric(row["class_index"], errors="coerce")
        timestamp = row["reviewed_at"]

        return (
            pd.notna(value)
            and float(value).is_integer()
            and 0 <= int(value) < len(self.label_set)
            and pd.notna(timestamp)
            and bool(str(timestamp).strip())
        )


    def scan_review_items(self, directory):
        directory = Path(directory).resolve()

        if not directory.is_dir():
            raise NotADirectoryError(directory)

        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        items = []
        keys = set()

        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in extensions:
                continue

            # Never follow a crop symlink outside the review directory.
            if path.is_symlink() or path.resolve().parent != directory:
                raise ValueError(f"Invalid review crop path: {path}")

            if not path.is_file():
                raise ValueError(f"Review crop is not a file: {path}")

            # Invalid image filenames deliberately stop loading/cleanup.
            item = parse_review_filename(path)
            key = self.review_key(item)

            if key in keys:
                raise ValueError(f"Duplicate review crop: {path.name}")

            keys.add(key)
            items.append(item)

        return items


    def _load_review_directory(self, directory):
        self._review_label_maps.clear()
        directory = normalize_directory(directory)
        items = self.scan_review_items(directory)
        table = self.read_review_table(directory)

        # Validate all current crops before changing the displayed dataset.
        for item in items:
            if cv2.imread(str(item.crop_path), cv2.IMREAD_UNCHANGED) is None:
                raise ValueError(f"Could not read crop: {item.crop_path}")
            self.ensure_review_row(table, item)

        if items:
            self.write_csv_atomic(table, directory / "for_review.csv")

        self.clear_state()
        self.loaded_review_directory = directory
        self.review_directory_input.setText(str(directory))
        self.review_items = items
        self.review_table = table
        self.rebuild_review_lookup()

        dropdown = self.review_item_counter_dropdown
        dropdown.blockSignals(True)
        try:
            dropdown.clear()
            dropdown.addItems([str(i + 1) for i in range(len(items))])
            dropdown.setCurrentIndex(-1)
        finally:
            dropdown.blockSignals(False)

        self.review_item_total_count_label.setText(f"/ {len(items)}")
        self.update_review_completion_status()

        if not items:
            QMessageBox.information(
                self, "No Review Items",
                "No review crops found. Existing CSV records were retained.",
            )
            return

        # Resume at the first pending crop.
        first_index = next(
            (
                i for i, item in enumerate(items)
                if self.review_key(item) not in self.completed_review_keys
            ),
            0,
        )

        self.switch_to_review_item(first_index)

        # Also handles reopening after an interrupted cleanup.
        self.finish_review_if_complete()


    def set_review_selection(self, class_index=None):
        self.labels_group.setExclusive(False)
        try:
            for button in self.labels_group.buttons():
                button.setChecked(False)
        finally:
            self.labels_group.setExclusive(True)

        if class_index is None:
            return

        name = self.label_set.name_for(int(class_index))
        for button in self.labels_group.buttons():
            if button.text() == name:
                button.setChecked(True)
                break


    def switch_to_review_item(self, index, save_previous=False):
        # Review decisions are saved by label clicks/number keys,
        # never implicitly by navigation.
        if not 0 <= index < len(self.review_items):
            return

        item = self.review_items[index]
        crop = cv2.imread(str(item.crop_path), cv2.IMREAD_UNCHANGED)
        if crop is None:
            raise ValueError(f"Could not read crop: {item.crop_path}")

        row_index = self.review_row_lookup.get(self.review_key(item))
        if row_index is None:
            raise ValueError(f"No CSV record for {item.crop_path.name}")

        row = self.review_table.loc[row_index]
        selection = (
            int(row["class_index"])
            if self.review_row_is_complete(row)
            else None
        )
        self.current_review_index = index
        self.current_review_item = item
        self.current_review_crop = crop
        self.image_display.setPixmap(self.review_crop_pixmap(item, crop))
        self.set_review_selection(selection)

        dropdown = self.review_item_counter_dropdown
        dropdown.blockSignals(True)
        try:
            dropdown.setCurrentIndex(index)
        finally:
            dropdown.blockSignals(False)

        self.review_image_id_label.setText(f"Image ID: {item.image_id}")
        self.review_cocoon_instance_label.setText(
            f"Cocoon: {item.instance_id}"
        )
        self.update_review_completion_status()
        self.setFocus()


    def update_review_item_from_dropdown(self, index):
        if self.mode != AppMode.REVIEW or index < 0:
            return

        previous_index = self.current_review_index

        try:
            self.switch_to_review_item(index)
        except Exception as error:
            dropdown = self.review_item_counter_dropdown
            dropdown.blockSignals(True)
            try:
                dropdown.setCurrentIndex(previous_index)
            finally:
                dropdown.blockSignals(False)

            self.show_error("Review Error", "Could not open review item.", error)


    def move_review_to_next_item(self, step):
        if not self.review_items:
            return

        index = self.current_review_index + step

        if 0 <= index < len(self.review_items):
            self.update_review_item_from_dropdown(index)
            return

        if step > 0:
            try:
                if self.finish_review_if_complete():
                    return
            except Exception as error:
                self.show_error(
                    "Review Error", "Could not finish review.", error
                )
                return

            QMessageBox.information(
                self, "Review",
                "This is the last review item. Some items still need review.",
            )


    def save_current_review_classification(self):
        item = self.current_review_item
        selected_class = self.selected_classification()

        if item is None or selected_class is None:
            return False

        if not item.crop_path.is_file():
            raise FileNotFoundError(item.crop_path)

        key = self.review_key(item)
        row_index = self.review_row_lookup.get(key)
        if row_index is None:
            raise ValueError(f"No CSV record for {item.crop_path.name}")
        metadata_path = self.loaded_review_directory / "for_review.csv"
        class_index = self.label_set.index_of(selected_class)

        # Invalidate a previous completion before changing original outputs.
        # If an output fails, the crop remains pending and available for retry.
        pending_table = self.review_table.copy()
        pending_table.at[row_index, "reviewed_at"] = pd.NA
        # Persist before changing labels so interrupted saves remain recoverable.
        pending_table.loc[
            pending_table["image_id"] == item.image_id, "overlay_pending"
        ] = 1

        try:
            self.write_csv_atomic(pending_table, metadata_path)
            self.review_table = pending_table
            self.completed_review_keys.discard(key)
            self.apply_review_classification_to_annotation_image(
                item, class_index
            )

            completed_table = self.review_table.copy()
            completed_table.at[row_index, "class_index"] = class_index
            completed_table.at[row_index, "reviewed_at"] = (
                datetime.now().astimezone().isoformat()
            )
            self.write_csv_atomic(completed_table, metadata_path)
            self.review_table = completed_table
            self.completed_review_keys.add(key)
        except Exception:
            self.review_image_cache = None
            self.set_review_selection()
            self.update_review_completion_status()
            raise

        self.update_review_completion_status()
        return True


    def rebuild_review_lookup(self):
        active_keys = {self.review_key(item) for item in self.review_items}
        lookup = {}
        completed = set()
        rows = self.review_table[REVIEW_KEY_COLUMNS].itertuples(
            index=False, name=None
        )
        for row_index, values in zip(self.review_table.index, rows):
            key = (str(values[0]), *(int(value) for value in values[1:]))
            if key not in active_keys:
                continue
            if key in lookup:
                raise ValueError(f"Duplicate review records for {key}")
            lookup[key] = row_index
            if self.review_row_is_complete(self.review_table.loc[row_index]):
                completed.add(key)
        missing = active_keys - lookup.keys()
        if missing:
            raise ValueError(f"{len(missing)} current crops have no review record.")
        self.review_row_lookup = lookup
        self.completed_review_keys = completed

    def review_progress(self):
        return len(self.completed_review_keys), len(self.review_items)


    def update_review_completion_status(self):
        completed, total = self.review_progress()
        self.review_item_total_count_label.setText(f"/ {total}")

        if total == 0:
            text = "Review status: No items"
            style = "color: black; font-weight: normal;"
        elif completed == total:
            text = f"Review status: Complete ({completed}/{total})"
            style = "color: green; font-weight: bold;"
        else:
            text = f"Review status: {completed}/{total}"
            style = "color: black; font-weight: normal;"

        self.review_completion_label.setText(text)
        self.review_completion_label.setStyleSheet(style)


    def finish_review_if_complete(self):
        directory = self.loaded_review_directory
        if directory is None:
            return False
        completed, total = self.review_progress()
        if total == 0 or completed != total:
            return False

        items = self.scan_review_items(directory)
        if not items:
            return False

        table = self.read_review_table(directory)

        for item in items:
            index = self.review_row_index(table, item)
            if index is None or not self.review_row_is_complete(table.loc[index]):
                return False

        self.review_completion_label.setText("Review status: Saving overlays...")
        self.review_completion_label.repaint()
        try:
            self.save_pending_review_overlays(table)
        except Exception:
            self.review_completion_label.setText(
                "Review status: Overlay save failed - retry to finish"
            )
            raise

        # Detect additions/removals while validating completion.
        current_items = self.scan_review_items(directory)
        if {item.crop_path for item in current_items} != {
            item.crop_path for item in items
        }:
            raise RuntimeError(
                "Review folder changed. Reload it before finishing."
            )

        failures = []
        for item in items:
            try:
                item.crop_path.unlink()
            except OSError as error:
                failures.append(f"{item.crop_path.name}: {error}")

        if failures:
            self.review_completion_label.setText(
                "Review status: Cleanup incomplete — reload directory"
            )
            raise OSError(
                "Some crops could not be deleted. CSV records were retained. "
                "Reload the review directory to retry:\n"
                + "\n".join(failures)
            )

        self.review_items = []
        self.current_review_item = None
        self.current_review_index = -1

        dropdown = self.review_item_counter_dropdown
        self.review_row_lookup = {}
        self.completed_review_keys = set()
        self.review_image_cache = None
        dropdown.blockSignals(True)
        try:
            dropdown.clear()
        finally:
            dropdown.blockSignals(False)

        self.image_display.clear()
        self.set_review_selection()
        self.review_image_id_label.setText("Image ID: ")
        self.review_cocoon_instance_label.setText("Cocoon: ")
        self.review_item_total_count_label.setText("/ 0")
        self.review_completion_label.setText("Review status: Complete")
        self.review_completion_label.setStyleSheet(
            "color: green; font-weight: bold;"
        )

        QMessageBox.information(
            self, "Review Complete",
            "All current review items were applied.\n"
            "Review crops were deleted; CSV records were retained.",
        )
        return True


    def save_pending_review_overlays(self, table):
        """Write each affected image once, retaining pending flags on failure."""
        pending_ids = sorted(set(table.loc[
            table["overlay_pending"].fillna(0).eq(1), "image_id"
        ]))
        # Final overlays must reflect the persisted labels, including retries.
        self.review_image_cache = None
        for image_id in pending_ids:
            cache = self.get_review_image_cache(image_id)
            crops = crop_instances(
                cache["image"], cache["label_image"], padding=10
            )
            overlay = render_completed_image_overlay(
                cache["image"], crops, cache["classifications"],
                self.label_set, alpha=0.15,
            )
            self.write_image_atomic(overlay, cache["overlay_path"])
            updated = table.copy()
            updated.loc[updated["image_id"] == image_id, "overlay_pending"] = 0
            self.write_csv_atomic(
                updated, self.loaded_review_directory / "for_review.csv"
            )
            table = updated
            self.review_table = updated

    def get_review_image_cache(self, image_id):
        root = self.loaded_review_directory.parent
        cache_key = (root, image_id)
        if self.review_image_cache is not None:
            if self.review_image_cache["key"] == cache_key:
                return self.review_image_cache
        self.review_image_cache = None
        recover_commit(root / ".import.pending.json")
        images = load_images_from_directory(root / "images")
        if image_id in images:
            recover_commit(self.annotation_journal(images[image_id]))
        labels = load_labels_from_directory(root / "labels")

        image_path = images.get(image_id)
        label_path = labels.get(image_id)

        if image_path is None:
            raise FileNotFoundError(
                f"Original image missing: {image_id}"
            )
        if label_path is None:
            raise FileNotFoundError(
                f"Original label map missing: {image_id}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        if image is None or label_image is None:
            raise ValueError(
                f"Could not read original image/label map: {image_id}"
            )

        if label_image.ndim == 3:
            label_image = label_image[:, :, 0]

        if image.shape[:2] != label_image.shape[:2]:
            raise ValueError(
                f"Image and label map sizes differ: {image_id}"
            )

        classification_path = root / "labels" / f"{image_id}.csv"
        if not classification_path.exists():
            raise FileNotFoundError(classification_path)

        classifications = validate_classifications(
            pd.read_csv(classification_path), len(self.label_set)
        )
        required = {"instance_id", "class_index"}
        if not required.issubset(classifications.columns):
            raise ValueError(
                f"Invalid annotation CSV: {classification_path}"
            )

        classifications["instance_id"] = pd.to_numeric(
            classifications["instance_id"], errors="raise"
        ).astype("Int64")
        classifications["class_index"] = pd.to_numeric(
            classifications["class_index"], errors="raise"
        ).astype("Int64")

        if classifications["instance_id"].isna().any():
            raise ValueError("Annotation CSV contains missing instance IDs.")

        if classifications["instance_id"].duplicated().any():
            raise ValueError("Annotation CSV contains duplicate instance IDs.")

        cache = {
            "key": cache_key,
            "image": image,
            "label_image": label_image,
            "label_path": label_path,
            "classification_path": classification_path,
            "overlay_path": root / "overlays" / f"{image_id}_class_overlay.png",
            "classifications": classifications,
            "instance_ids": set(np.unique(label_image)) - {0},
        }
        self.review_image_cache = cache
        return cache

    def apply_review_classification_to_annotation_image(self, item, class_index):
        cache = self.get_review_image_cache(item.image_id)
        classifications = cache["classifications"].copy()
        instance_ids = cache["instance_ids"].copy()
        label_image = cache["label_image"]
        exclude = self.label_set.name_for(class_index) == EXCLUDE_LABEL
        row_mask = classifications["instance_id"] == item.instance_id

        if exclude:
            # Idempotent: retrying an already-applied exclusion is allowed.
            label_image = label_image.copy()
            label_image[label_image == item.instance_id] = 0
            classifications = classifications.loc[~row_mask].copy()
            instance_ids.discard(item.instance_id)
        else:
            if item.instance_id not in instance_ids:
                raise ValueError(
                    f"Instance {item.instance_id} is absent from "
                    f"{cache['label_path'].name}. It may already have been excluded."
                )

            if row_mask.any():
                classifications.loc[row_mask, "class_index"] = class_index
            else:
                classifications = pd.concat(
                    [
                        classifications,
                        pd.DataFrame([{
                            "instance_id": item.instance_id,
                            "class_index": class_index,
                        }]),
                    ],
                    ignore_index=True,
                )

        # Normal class changes preserve segmentation instance IDs.
        # Exclude changes both the segmentation and classification CSV.
        try:
            if exclude:
                self.write_image_atomic(label_image, cache["label_path"])
            self.write_csv_atomic(classifications, cache["classification_path"])
        except Exception:
            self.review_image_cache = None
            raise

        cache["label_image"] = label_image
        cache["classifications"] = classifications
        cache["instance_ids"] = instance_ids



    def show_error(self, title:str, message:str, error:Exception)->None:
        logger.exception("%s: %s", title, message)
        QMessageBox.critical(self, title, f"{message}\n\nError: {error}")

    def create_annotation_shortcut_bar(self):
        shortcuts = [
            ((["\u2190"], "", "Previous cocoon"), (["Ctrl", "\u2190"], "+", "Previous image")),
            ((["0", "7"], "-", "Choose label"), (["Space"], "", "Show/hide overlay")),
            ((["\u2192"], "", "Next cocoon"), (["Ctrl", "\u2192"], "+", "Next image")),
        ]

        return self.create_shortcut_bar_from_shortcuts(shortcuts)

    def create_review_shortcut_bar(self):
        shortcuts = [
            ((["\u2190"], "", "Previous review item"), None),
            ((["0", "7"], "-", "Choose label"), (["Space"], "", "Show/hide overlay")),
            ((["\u2192"], "", "Next review item"), None),
        ]

        return self.create_shortcut_bar_from_shortcuts(shortcuts)

    def create_shortcut_bar_from_shortcuts(self, shortcuts):
        shortcut_container = QWidget()
        shortcut_container.setObjectName("shortcutBar")
        shortcut_container.setFixedHeight(58)

        shortcut_layout = QGridLayout(shortcut_container)
        shortcut_layout.setContentsMargins(10, 3, 10, 3)
        shortcut_layout.setHorizontalSpacing(28)
        shortcut_layout.setVerticalSpacing(1)

        def shortcut_item(keys, separator, action):
            item = QWidget()
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)
            item_layout.setAlignment(Qt.AlignCenter)

            key_group = QWidget()
            key_group_layout = QHBoxLayout(key_group)
            key_group_layout.setContentsMargins(0, 0, 0, 0)
            key_group_layout.setSpacing(4)

            for key_index, key in enumerate(keys):
                key_label = QLabel(key)
                key_label.setFixedHeight(18)
                key_label.setMinimumWidth(24)
                key_label.setAlignment(Qt.AlignCenter)
                key_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: #f7f7f7;
                        border: 1px solid #aaaaaa;
                        border-bottom: 2px solid #888888;
                        border-radius: 3px;
                        padding: 1px 5px;
                        font-size: 9px;

                    }
                    """
                )

                key_group_layout.addWidget(key_label)

                if key_index < len(keys) - 1:
                    separator_label = QLabel(separator)
                    separator_label.setAlignment(Qt.AlignCenter)
                    separator_label.setStyleSheet(
                        """
                        QLabel {
                            color: #555555;
                            border: none;
                            background: transparent;
                            font-size: 9px;
                        }
                        """
                    )
                    key_group_layout.addWidget(separator_label)

            action_label = QLabel(action)
            action_label.setStyleSheet(
                """
                QLabel {
                    color: #222222;
                    background: transparent;
                    border: none;
                    font-size: 9px;
                }
                """
            )

            item_layout.addWidget(key_group)
            item_layout.addWidget(action_label)
            return item

        for column, (top_shortcut, bottom_shortcut) in enumerate(shortcuts):
            shortcut_layout.addWidget(
                shortcut_item(*top_shortcut),
                0,
                column,
                alignment=Qt.AlignCenter,
            )
            if bottom_shortcut is None:
                continue
            shortcut_layout.addWidget(
                shortcut_item(*bottom_shortcut),
                1,
                column,
                alignment=Qt.AlignCenter,
            )

        shortcut_layout.setColumnStretch(0, 1)
        shortcut_layout.setColumnStretch(1, 1)
        shortcut_layout.setColumnStretch(2, 1)

        shortcut_container.setStyleSheet(
            """
            QWidget#shortcutBar {
                background-color: #eeeeee;
                border-top: 1px solid #c0c0c0;
            }
            """
        )

        return shortcut_container

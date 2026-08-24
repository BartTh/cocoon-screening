from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import QDialog, QButtonGroup, QGridLayout, QFileDialog, QHBoxLayout, QMainWindow, QLabel, QVBoxLayout, QWidget,QMessageBox, QPushButton, QLineEdit, QComboBox, QCheckBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
import cv2
import logging
import numpy as np
import pandas as pd
from src.cocoon_screening.processing.stardist import StardistPredictor
from src.cocoon_screening.io.data import get_batch_id, load_images_from_directory, load_labels_from_directory
from src.cocoon_screening.paths import bundled_configs_dir, default_images_dir
from src.cocoon_screening.ui.utils import array_to_pixmap
from src.cocoon_screening.processing.image_processing import draw_instance_overlay, render_completed_image_overlay, crop_instances
from src.cocoon_screening.domain.labels import load_labels
from src.cocoon_screening.ui.dialogs.import_xray_dialog import ImportXrayDialog
from src.cocoon_screening.io.batch_importer import import_batch

EXCLUDE_LABEL = "Exclude"
UNSURE_LABEL = "Other (Unsure)"
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.stardist_predictor = StardistPredictor(use_custom_model=True)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cocoon Annotation Tool")

        ## Main layout 
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)

        ## Variables
        labels_path = bundled_configs_dir() / "labels.yaml"
        self.label_set = load_labels(labels_path)
        self.images_path_dict = {}
        self.labels_path_dict = {}
        self.current_crops = {}
        self.current_classifications = pd.DataFrame(columns=["instance_id", "class_index"])
        self.current_cocoon_id = None
        self.current_image = None

        ## Directory input
        directory_container = QWidget()
        directory_layout = QHBoxLayout(directory_container)
        
        directory_label = QLabel("Image Directory:")
        directory_layout.addWidget(directory_label)

        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText("Enter directory path...")
        self.directory_input.returnPressed.connect(lambda: self.load_directory(self.directory_input.text()))
        directory_layout.addWidget(self.directory_input)

        dir_input_button = QPushButton("Browse...")
        dir_input_button.clicked.connect(self.browse_directory)
        directory_layout.addWidget(dir_input_button)

        main_layout.addWidget(directory_container)

        ## Import XRay button
        or_label = QLabel("or")
        main_layout.addWidget(or_label, alignment=Qt.AlignCenter)

        import_button = QPushButton("Import X-rays")
        import_button.clicked.connect(self.open_import_xray_dialog)
        main_layout.addWidget(import_button, alignment=Qt.AlignCenter)

        ## Status bar
        status_container = QWidget()
        status_container.setObjectName("status_container")
        status_container.setStyleSheet(
            """
            QWidget#status_container {
                background-color: #f7f7f7;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            """
        )
        status_layout = QHBoxLayout(status_container)
        status_layout.setAlignment(Qt.AlignLeft)

        # batch ID label
        self.batch_id_label = QLabel("Batch ID: ") 
        status_layout.addWidget(self.batch_id_label)

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
        status_layout.addWidget(image_count_container)

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
        status_layout.addWidget(cocoon_count_container)

        self.image_completion_label = QLabel("Image status: ")
        status_layout.addWidget(self.image_completion_label)

        main_layout.addWidget(status_container)

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
        shortcut_container = self.create_shortcut_bar()
        main_layout.addWidget(shortcut_container)

        # Keyboard shortcuts
        self.create_shortcuts()
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure the main window can receive keyboard events

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

            QMessageBox.information(
                self,
                "Import Complete", 
                f"Batch {result.batch_id} successfully imported!"
            )

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
            self.load_directory(str(result.output_directory))

        except Exception as e:
            self.show_error("Import Error", f"Failed to import batch {request.batch_id}:\n{e}", e)
            
    def display_cocoon(self, text):
        if not str(text).isdigit():
            return
        new_instance_id = int(text)
        if new_instance_id not in self.current_crops:
            logger.warning("Invalid cocoon index %s for image %s", new_instance_id, self.current_image.stem)
            return

        self.current_cocoon_id = new_instance_id

        crop = self.current_crops[new_instance_id]
        restored = self.restore_saved_classification_selection()

        overlay_image = draw_instance_overlay(
            crop["image"],
            crop["mask"],
            color=(0,255,182),
            alpha=0.35,
        )

        pixmap = array_to_pixmap(overlay_image)
        
        DISPLAY_SCALE = 2
        new_width = int(pixmap.width() * DISPLAY_SCALE)
        new_height = int(pixmap.height() * DISPLAY_SCALE)
        
        scaled_pixmap = pixmap.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_display.setPixmap(scaled_pixmap)
        self.cocoon_count_label.setText(f"Cocoon {crop['instance_id']}: ")

        # Select the first label by default
        if not restored:
            buttons = self.labels_group.buttons()
            if buttons:
                buttons[0].setChecked(True)

    def selected_classification(self):
        selected_button = self.labels_group.checkedButton()
        if selected_button:
            return selected_button.text()
        return None
    
    def classification_file_path(self):
        directory = Path(self.directory_input.text())
        return directory.parent / "labels" / f"{self.current_image.stem}.csv"
    
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

        # Label selection shortcuts
        for i in range(min(len(self.label_set), 9)):
            label_shortcut = QShortcut(QKeySequence(f"{i}"), self)
            label_shortcut.activated.connect(lambda idx=i: self.select_label_by_index(idx))

    def select_label_by_index(self, index):
        if not 0 <= index < len(self.label_set):
            return
        label_name = self.label_set.names[index]
        for button in self.labels_group.buttons():
            if button.text() == label_name:
                button.setChecked(True)
                self.on_label_selected()
                return
            
    def current_image_is_fully_classified(self):
        if self.current_classifications.empty:
            return False

        if len(self.current_classifications) != len(self.current_crops):
            return False

        return self.current_classifications["class_index"].notna().all()
    
    def update_image_completion_status(self):
        if self.current_image_is_fully_classified():
            self.image_completion_label.setText("Image status: Complete")
            self.image_completion_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            labelled_count = self.current_classifications["class_index"].notna().sum()
            total_count = len(self.current_crops)
            self.image_completion_label.setText(
                f"Image status: {labelled_count}/{total_count}"
            )
            self.image_completion_label.setStyleSheet(
                "color: black; font-weight: normal;"
            )

    def next_cocoon(self):
        current_index = self.cocoon_idx_dropdown.currentIndex()
        if current_index < len(self.current_crops) - 1:
            self.cocoon_idx_dropdown.setCurrentIndex(current_index + 1)
        else:
            # If at the last cocoon, move to the next image
            self.next_image()

    def prev_cocoon(self):
        current_index = self.cocoon_idx_dropdown.currentIndex()
        if current_index > 0:
            self.cocoon_idx_dropdown.setCurrentIndex(current_index - 1)
        else:
            # If at the first cocoon, move to the previous image
            self.prev_image()
    
    def next_image(self):
        current_index = self.image_counter_dropdown.currentIndex()
        if current_index < len(self.images_path_dict) - 1:
            self.image_counter_dropdown.setCurrentIndex(current_index + 1)
        else:
            QMessageBox.information(self, "Navigation", "There is no next image.")

    def prev_image(self):
        current_index = self.image_counter_dropdown.currentIndex()
        if current_index > 0:
            self.image_counter_dropdown.setCurrentIndex(current_index - 1)
        else:
            QMessageBox.information(self, "Navigation", "There is no previous image.")
    
    def on_label_selected(self):
        try:
            self._on_label_selected()
        except Exception as error:
            self.show_error("Error", "The selected class could not be applied.", error)

    def _on_label_selected(self):
        current_classification = self.selected_classification()
        if current_classification == EXCLUDE_LABEL:
            self.exclude_current_cocoon()
            return
        
        if current_classification == UNSURE_LABEL:
            self.save_current_cocoon_for_review()

        self.save_current_cocoon_classification()
        self.next_cocoon()

    def for_review_directory(self):
        image_directory = Path(self.directory_input.text())
        return image_directory.parent / "for_review"
    
    def label_map_file_path(self):
        image_id = Path(self.current_image).stem
        labels_directory = Path(self.current_image).parent.parent / "labels"

        return labels_directory / f"{image_id}.png"

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

        crop_image = crop["image"]
        scaled_crop = cv2.resize(
            crop_image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_LINEAR,
        )

        if not cv2.imwrite(str(crop_path), scaled_crop):
            raise IOError(
                f"Failed to save cocoon crop of {instance_id} of image {image_id} "
                f"for review to {crop_path}"
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
                    "class_index": self.label_set.index_of(UNSURE_LABEL),
                }
            ]
        )

        if metadata_path.exists():
            review_table = pd.read_csv(metadata_path)
            duplicate_mask = (
                (review_table["image_id"] == image_id) & 
                (review_table["instance_id"] == instance_id)
            )
            if duplicate_mask.any():
                review_table.loc[duplicate_mask, :] = row.iloc[0]
            else:
                review_table = pd.concat(
                    [review_table, row],
                    ignore_index=True,
                )
        else:
            review_table = row

        review_table.to_csv(
            metadata_path,
            index=False,
        )
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
            self.show_error("Error", "The cocoon could not be removed from the label map.", error)

    def _exclude_current_cocoon(self):
        if self.current_cocoon_id is None:
            return
        instance_id = self.current_cocoon_id
        logger.info(
            "cocoon instance excluded: image_id=%s, instance_id=%s, selected_label=%s",
            self.current_image.stem,
            instance_id,
            self.selected_classification(),
        )
        if instance_id not in self.current_crops:
            logger.warning("Invalid cocoon instance ID %s for image id %s", instance_id, self.current_image.stem)
            return
        next_instance_id = self.next_instance_id_after_removal(instance_id)

        self.remove_instance_from_label_image(instance_id)
        self.remove_instance_from_current_state(instance_id)
        self.remove_instance_from_label_csv(instance_id)
        self.remove_instance_from_dropdown(instance_id)

        self.update_image_completion_status()
        self.save_completed_image_overlay()

        if next_instance_id is not None:
            self.cocoon_idx_dropdown.blockSignals(True)
            self.cocoon_idx_dropdown.setCurrentText(str(next_instance_id))
            self.cocoon_idx_dropdown.blockSignals(False)
            self.display_cocoon(str(next_instance_id))
        else:
            self.current_cocoon_id = None
            self.cocoon_count_label.setText("Cocoon: ")
            self.image_display.clear()

    
    def remove_instance_from_label_image(self,instance_id):
        label_map_path = self.label_map_file_path()
        label_image = cv2.imread(str(label_map_path), cv2.IMREAD_UNCHANGED)

        if label_image is None:
            raise ValueError(f"Could not read label map image: {label_map_path}")

        if label_image.ndim == 3:
            label_image = label_image[:,:,0]

        label_image[label_image==instance_id]=0
        
        if not cv2.imwrite(str(label_map_path), label_image.astype(np.uint16)):
            raise IOError(f"Failed to update label map image after removing cocoon instance {instance_id} for image {self.current_image.stem}")

    def remove_instance_from_current_state(self,instance_id):
        self.current_crops.pop(instance_id, None)

        if self.current_cocoon_id == instance_id:
            self.current_cocoon_id = None
        
    def remove_instance_from_label_csv(self,instance_id):
        row_mask = self.current_classifications["instance_id"]==instance_id
        self.current_classifications = self.current_classifications.loc[~row_mask].copy()

        self.current_classifications.to_csv(self.classification_file_path(), index=False)
        
    def remove_instance_from_dropdown(self,instance_id):
        index = self.cocoon_idx_dropdown.findText(str(instance_id))
        if index==-1:
            return
        self.cocoon_idx_dropdown.blockSignals(True)
        self.cocoon_idx_dropdown.removeItem(index)
        self.cocoon_idx_dropdown.blockSignals(False)
        self.cocoon_total_count_label.setText(f"/ {len(self.current_crops)}")

    def update_image_id_from_dropdown(self, text):
        if not str(text).isdigit():
            return

        index = int(text) - 1  # Convert to 0-based index

        if 0 <= index < len(self.images_path_dict):
            self.switch_to_image(index, save_previous=True)

        self.setFocus()  # Set focus back to the main window to ensure keyboard shortcuts work

    def switch_to_image(self, index, save_previous=True):
        try:
            self._switch_to_image(index, save_previous)
        except Exception as error:
            self.show_error("Error", f"Failed to open image number {index+1}", error)

    def _switch_to_image(self, index, save_previous=True):
        if save_previous:
            self.save_current_cocoon_classification()

        directory = self.directory_input.text()
        image_id = list(self.images_path_dict.keys())[index]

        self.current_image = self.images_path_dict[image_id]
        self.current_cocoon_id = None

        self.image_id_label.setText(str(image_id))
        self.batch_id_label.setText(f"Batch ID: {get_batch_id(Path(directory), image_id)}")

        self.load_annotation_crops(image_id)
        self.ensure_classification_file_exists()
        self.update_image_completion_status()

        if self.current_crops:
            first_instance_id = next(iter(self.current_crops))

            self.cocoon_idx_dropdown.blockSignals(True)
            self.cocoon_idx_dropdown.setCurrentText(str(first_instance_id))
            self.cocoon_idx_dropdown.blockSignals(False)

            self.display_cocoon(str(first_instance_id))
        else:
            self.current_cocoon_id = None

        logger.info("Starting annotation of selected image: %s", image_id)

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
        self.save_current_cocoon_classification()
        self.display_cocoon(text)
        self.setFocus()  # Set focus back to the main window to ensure keyboard shortcuts work

    def save_completed_image_overlay(self) -> None:
        if self.current_image is None:
            return

        if not self.current_image_is_fully_classified():
            return
        self.update_image_completion_status()

        image_path = Path(self.current_image)
        try: 
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        except Exception:
            logger.exception("Could not read image to create overlay: %s", image_path)
            return

        result = render_completed_image_overlay(
            image,
            self.current_crops,
            self.current_classifications,
            self.label_set,
            alpha=0.15,
        )

        out_path = image_path.parent.parent / "overlays" / f"{image_path.stem}_class_overlay.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        if not cv2.imwrite(str(out_path), result):
            logger.error("Failed to save completed overlay for image: %s to %s", image_path, out_path)
            return

        logger.info(
            "All cocoons in image %s classified! Complete overlay saved: %s",
            image_path.stem, out_path,
        )

    def save_current_cocoon_classification(self):
        if self.current_cocoon_id is None:
            return
        if self.current_cocoon_id not in self.current_crops:
            logger.warning("Invalid cocoon index %s for image %s", self.current_cocoon_id, self.current_image.stem)
            return
        current_crop = self.current_crops[self.current_cocoon_id]
        instance_id = current_crop["instance_id"]
        selected_class = self.selected_classification()
        class_index = self.label_set.index_of(selected_class) if selected_class is not None else None
        current_crop["class_index"] = class_index
        row_mask = self.current_classifications["instance_id"] == instance_id
        if row_mask.any():
            self.current_classifications.loc[row_mask, "class_index"] = class_index
        else:
            self.current_classifications.loc[len(self.current_classifications)] = {
                "instance_id": instance_id,
                "class_index": class_index,
            }

        self.current_classifications.to_csv(self.classification_file_path(), index=False)
        self.update_image_completion_status()
        self.save_completed_image_overlay()

    def closeEvent(self, event):
        self.save_current_cocoon_classification()
        super().closeEvent(event)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory", str(default_images_dir()))
        if directory:
            self.load_directory(directory)

    def load_directory(self, directory):
        try:
            self._load_directory(directory)
        except Exception as error:
            self.show_error("Error", "Failed to load directory", error)

    def _load_directory(self, directory):
        self.directory_input.setText(directory)
        logger.info("directory_loaded | directory=%s",directory,)
        self.images_path_dict = load_images_from_directory(directory)
        self.labels_path_dict = load_labels_from_directory(Path(directory).parent / "labels")

        self.image_counter_dropdown.blockSignals(True)
        self.image_counter_dropdown.clear()
        self.image_counter_dropdown.addItems([str(i + 1) for i in range(len(self.images_path_dict))])
        self.image_counter_dropdown.setCurrentIndex(0)
        self.image_counter_dropdown.blockSignals(False)

        self.image_total_count_label.setText(f"/ {len(self.images_path_dict)}")

        if self.images_path_dict:
            self.switch_to_image(0, save_previous=False)

    def ensure_classification_file_exists(self):
        classification_file = self.classification_file_path()
        classification_file.parent.mkdir(parents=True, exist_ok=True)

        if not classification_file.exists():
            if self.current_crops:
                self.current_classifications = pd.DataFrame(self.current_crops.values())
            else:
                self.current_classifications = pd.DataFrame(columns=["instance_id", "class_index"])

            self.current_classifications = self.current_classifications[["instance_id", "class_index"]].copy()
            self.current_classifications["class_index"] = pd.to_numeric(self.current_classifications["class_index"], errors='coerce').astype('Int64')
            self.current_classifications.to_csv(classification_file, index=False)
        else:
            self.current_classifications = pd.read_csv(classification_file)
            self.current_classifications["class_index"] = pd.to_numeric(self.current_classifications["class_index"], errors='coerce').astype('Int64')

    def load_annotation_crops(self, image_id):
        label_path = self.labels_path_dict.get(image_id)

        # Create label if no labels
        if label_path is None:

            image_path = Path(self.current_image)

            labels_directory = (
                image_path.parent.parent / "labels"
            )

            label_path = (
                labels_directory / f"{image_id}.png"
            )

            self.stardist_predictor.create_label_map(
                image_path,
                label_path,
            )

            self.labels_path_dict[image_id] = label_path

        self.current_crops = {}

        image = cv2.imread(
            str(self.current_image),
            cv2.IMREAD_UNCHANGED,
        )

        label_image = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {self.current_image}"
            )

        if label_image is None:
            raise ValueError(
                f"Could not read label: {label_path}"
            )

        if label_image.ndim == 3:
            label_image = label_image[:, :, 0]

        if image.shape[:2] != label_image.shape[:2]:
            raise ValueError(
                f"Image and label sizes differ: "
                f"{image.shape[:2]} vs "
                f"{label_image.shape[:2]}"
            )

        self.current_crops = crop_instances(
            image,
            label_image,
            padding=10,
        )

        self.cocoon_idx_dropdown.blockSignals(True)

        self.cocoon_idx_dropdown.clear()
        
        self.cocoon_idx_dropdown.addItems(
            [
                str(instance_id)
                for instance_id in self.current_crops.keys()
            ]
        )

        self.cocoon_idx_dropdown.blockSignals(False)

        self.cocoon_total_count_label.setText(
            f"/ {len(self.current_crops)}"
        )

    def show_error(self, title:str, message:str, error:Exception)->None:
        logger.exception("%s: %s", title, message)
        QMessageBox.critical(self, title, f"{message}\n\nError: {error}")

    def create_shortcut_bar(self):
        shortcut_container = QWidget()
        shortcut_container.setObjectName("shortcutBar")
        shortcut_container.setFixedHeight(58)
        shortcut_layout = QGridLayout(shortcut_container)
        shortcut_layout.setContentsMargins(10, 3, 10, 3)
        shortcut_layout.setHorizontalSpacing(28)
        shortcut_layout.setVerticalSpacing(1)

        shortcuts = [
            ((["\u2190"], "", "Previous cocoon"), (["Ctrl", "\u2190"], "+", "Previous image")),
            ((["0", "7"], "-", "Choose label"), None),
            ((["\u2192"], "", "Next cocoon"), (["Ctrl", "\u2192"], "+", "Next image")),
        ]

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

# Cocoon Screening

The application detects individual cocoons with a custom StarDist instance-segmentation model and allows the user to assign a class label to each detected cocoon.

## Repository Structure

```text
cocoon-screening/
  main.py                         # Starts the GUI application
  requirements.txt                # Python dependencies
  configs/
    labels.yaml                   # Label names and colors used in the GUI
  models/
    bee_cocoon_stardist/          # Custom StarDist model weights
  data/
    raw/                          # Raw X-ray batch for testing
  scripts/
    import_batch.py               # Command-line batch import helper
  src/cocoon_screening/
    app.py                        # QApplication setup and main window startup
    paths.py                      # App paths for configs, models, logs, data defaults
    domain/                       # Labels, review items, validation
    io/                           # Data loading, batch import, logging
    processing/                   # StarDist prediction and image processing
    ui/                           # Main GUI, dialogs, and widgets
```

## Setup

Create an environment:

```powershell
conda create -n cocoons python=3.10.20
conda activate cocoons
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

## Build Windows Executable

The GUI can be packaged as a single Windows executable with PyInstaller:

```powershell
pyinstaller --windowed --onefile --name "CocoonAnnotationTool" --add-data "configs;configs" --add-data "models\bee_cocoon_stardist\config.json;models\bee_cocoon_stardist" --add-data "models\bee_cocoon_stardist\weights_best.h5;models\bee_cocoon_stardist" main.py
```

The bundled executable is written to:

```text
dist/CocoonAnnotationTool.exe
```

## How to Use


You can either start the application using the code by running the following from the repository root:
   ```powershell
   python main.py
   ```
OR run the application using the executable.

There are two modes in the application: Annotation and Review Mode. The former is the standard data annotation while the latter is a special mode to iterate through uncertain cocoon instances marked for review  and relabel them. You can switch between the modes by using the radio buttons.
### Annotation Mode
1. Choose an existing image directory with **Browse...** or use **Import X-rays** to import a raw batch into an output image directory.
   
   To import, the following values are needed:
   - **Source directory**: raw batch folder, for example `C:/data/raw/batch_0001`
   - **Output directory**: processed image folder, for example `C:/data/images`
   - **Batch ID**: batch name, for example `C:/batch_0001`

   The same import can be run from the command line:

   ```powershell
    python -m scripts.import_batch --batch-id "batch_0001" --batches-directory "data/raw" --output-directory "data/images"
   ```

2. The first image from the directory is loaded automatically. When an image is first opened, the app looks for a matching label map in `../labels/<image_id>.png`. If it does not exist, the custom StarDist model creates one automatically. If a classification file exists, it is loaded along with its existing instance classifications and else a new empty file is created in `../labels/<image_id>.csv`.

3. The first detected cocoon is displayed and class is automatically set to `Prepupae` per default. If an existing classification is restored, the corresponding label will be displayed instead.

4. Choose another label with the checkbox panel or keyboard number.
   
   If a cocoon is marked as `Other (Unsure)`, a scaled crop and metadata row are saved to:

    ```text
    ../for_review/
    ```

   If a cocoon is marked as `Exclude`, that instance is removed from the label map and from the classification CSV and also from the dropdown list.

5. After selecting the class OR when navigating to another cocoon by Arrow Keys or the dropdown menu OR when closing, the classification is saved immediately to the corresponding classification file and the GUI moves on to the next cocoon.
6. When the image is complete and all cocoon instances are classified, an overlay with all instances colored in their class label is written to:
   ```text
   ../overlays/<image_id>_class_overlay.png
   ```

### Review Mode
1. If an images directory is already loaded in the annotation mode, the application automatically suggests its sibling `for_review` directory. You can also browse to another review directory or enter the path manually and press Enter.

2. All cocoon instances in the `for_review` directory will be listed as review items. The active review queue is based only on crop files currently present. Historical CSV entries without a corresponding crop do not count toward the active review progress.
    
3. The first pending review item in the list will be displayed alongside its corresponding image ID and cocoon instance ID. Previously completed review decisions are restored automatically. Pending review items have no class selected.

4. Choose a class using the label panel or keyboard shortcut. The decision updates the original classification CSV.

5. After the decision is saved, the application automatically moves to the next review item. The Left and Right arrow keys or the dropdown menu can be used to move between review items without assigning a class.

6. Review progress and history of past review sessions are stored in:

   ```text
   ../for_review/for_review.csv
   ```

    The CSV contains the following fields:

    | Field | Meaning |
    |---|---|
    | `image_id` | Original image identifier |
    | `instance_id` | Cocoon instance identifier within the image |
    | `x1`, `y1`, `x2`, `y2` | Crop bounding-box coordinates |
    | `crop_path` | Recorded location of the review crop |
    | `original_class_index` | Class index recorded when queued for review |
    | `class_index` | Class index assigned during review |
    | `reviewed_at` | Timestamp recorded after the label changes succeed |
    | `overlay_pending` | Whether the image’s class overlay needs regeneration |

7. When all current review items have been reviewed, the application:

   - rechecks the review directory and metadata
   - regenerates pending overlays once per affected original image
   - clears the corresponding `overlay_pending` flags after successful writes
   - removes the completed crop images from `for_review`
   - keeps `for_review.csv` as the review history

## Keyboard Shortcuts

```text
Left              Previous cocoon
Right             Next cocoon
Ctrl + Left       Previous image
Ctrl + Right      Next image
0-7               Select class label by index
```

## Outputs and Logs

The app writes label outputs, overlays, and review crops next to the selected image directory, so it is recommended to create a data folder:

```text
data/
  raw/                # Raw X-ray files
  images/             # Imported JPG images
  labels/             # Label maps and classification CSV files
  overlays/           # Completed class color overlays
  for_review/         # Unclear cocoons that need to be reviewed
  image_registry.csv  # Maps image_id to batch_id and original DICOM name
```

Application logs are written to the current user's application-data directory:

- Windows: `%LOCALAPPDATA%/CocoonScreening/logs/` (with an `AppData/Local` fallback).
- macOS: `~/Library/Application Support/CocoonScreening/logs/`.
- Linux: `$XDG_DATA_HOME/CocoonScreening/logs/` or `~/.local/share/CocoonScreening/logs/`.


## Notes
- IMPORTANT: If X-rays are imported, image IDs are continued in the format `XR_000001` from the last image ID read in the `image_registry.csv`.
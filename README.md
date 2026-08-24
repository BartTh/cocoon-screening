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
1. You can either start the application using the code by running the following from the repository root:
   ```powershell
   python main.py
   ```
   OR run the application using the executable.

2. Choose an existing image directory with **Browse...** or use **Import X-rays** to import a raw batch into an output image directory.
   
   To import, the following values are needed:
   - **Source directory**: raw batch folder, for example `data/raw/batch_0001`
   - **Output directory**: processed image folder, for example `data/images`
   - **Batch ID**: batch name, for example `batch_0001`

   The same import can be run from the command line:

   ```powershell
    python -m scripts.import_batch --batch-id "batch_0001" --batches-directory "data/raw" --output-directory "data/images"
   ```

3. The first image from the directory is loaded automatically. When an image is first opened, the app looks for a matching label map in `../labels/<image_id>.png`. If it does not exist, the custom StarDist model creates one automatically. If a classification file exists, it is loaded along with its existing instance classifications and else a new empty file is created in `../labels/<image_id>.csv`.

4. The first detected cocoon is displayed and class is automatically set to `Prepupae`.

5. Choose another label with the checkbox panel or keyboard number.
   
   If a cocoon is marked as `Other (Unsure)`, a scaled crop and metadata row are saved to:

    ```text
    ../for_review/
    ```

   If a cocoon is marked as `Exclude`, that instance is removed from the label map and from the classification CSV and also from the dropdown list.

6. After selecting the class, the classification is saved immediately to the corresponding classification file and the GUI moves on to the next cocoon.
7. When the image is complete and all cocoon instances are classified, an overlay with all instances colored in their class label is written to:
   ```text
   ../overlays/<image_id>_class_overlay.png
   ```


## Keyboard Shortcuts

```text
Left              Previous cocoon
Right             Next cocoon
Ctrl + Left       Previous image
Ctrl + Right      Next image
0-7               Select class label by index
```

## Outputs and Logs

The app writes annotation outputs next to the selected image directory, so it is recommended to create a data folder:

```text
data/
  raw/                # Raw X-ray files
  images/             # Imported JPG images
  labels/             # Label maps and classification CSV files
  overlays/           # Completed class color overlays
  for_review/         # Unclear cocoons that need to be reviewed
  image_registry.csv  # Maps image_id to batch_id and original DICOM name
```

Application logs are written under the local user data directory:

```text
%LOCALAPPDATA%\CocoonScreening\logs\
```


## Notes
- IMPORTANT: If X-rays are imported, image IDs are continued in the format `XR_000001` from the last image ID read in the `image_registry.csv`.

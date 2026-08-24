# add argparse to run the script from the command line

import argparse
from pathlib import Path
import logging
from src.cocoon_screening.io.logging import setup_logging
from src.cocoon_screening.paths import user_log_path
from src.cocoon_screening.io.batch_importer import import_batch

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Import DICOM X-ray images batch")
    parser.add_argument("--batch-id", required=True, help="Batch ID to import (e.g., batch_0001)")
    parser.add_argument("--output-format", default="jpg", help="Output image format (default: jpg)")
    parser.add_argument("--batches-directory", default="data/raw", help="Directory containing batch folders (default: data/raw)")
    parser.add_argument("--output-directory", default="data/images", help="Directory to save processed images (default: data/output)")

    args = parser.parse_args()

    import_batch(
        batch_directory=Path(args.batches_directory) / args.batch_id,
        output_directory=Path(args.output_directory),
        image_registry_path=Path(args.output_directory).parent / "image_registry.csv",
        batch_id=args.batch_id,
        output_format=args.output_format
    )

    logger.info(f"Batch {args.batch_id} successfully imported to {args.output_directory}!")

if __name__ == "__main__":
    setup_logging(user_log_path(), log_level=logging.INFO)
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during batch import: %s", e)
        raise SystemExit(1)
# python -m  scripts.import_batch --batch-id batch_0001 
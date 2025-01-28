import logging
import sys


def test_installation():
    # List of required packages
    packages = {"pandas", "numpy", "sklearn"}
    missing_packages = []

    # Check if the required packages are installed
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    # Logging the result
    if missing_packages:
        logging.error("Some required packages are missing:")
        for package in missing_packages:
            logging.error(f"- {package}")
        return False
    else:
        logging.info("Installation test completed successfully!")
        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Run the test and exit with status code 1 if any package is missing
    if not test_installation():
        sys.exit(1)  # Exit with error if any package is missing

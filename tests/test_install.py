import logging
import sys


def test_installation():
    # List of required packages
    packages = {"pandas", "numpy", "sklearn", "pytest", "matplotlib"}
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
        assert (
            False
        ), f"Missing packages: {', '.join(missing_packages)}"  # Assert failure if packages are missing
    else:
        logging.info("Installation test completed successfully!")
        assert True  # Assert success if all packages are found


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Run the test
    test_installation()  # No need to call sys.exit here, as pytest handles exit codes

import os
import subprocess
import sys

def ensure_sqlite_version():
    # Define paths
    sqlite_dir = os.path.expanduser("~/sqlite")  # Directory to store local SQLite
    os.makedirs(sqlite_dir, exist_ok=True)

    # Path to the SQLite binary
    sqlite_bin = os.path.join(sqlite_dir, "bin", "sqlite3")

    # Check current SQLite version if the local binary exists
    if os.path.exists(sqlite_bin):
        sqlite_version = subprocess.getoutput(f"{sqlite_bin} --version")
        print(f"Current SQLite version: {sqlite_version}")
    else:
        sqlite_version = "0.0.0"  # Assume it's not installed locally

    # Upgrade SQLite if necessary
    if "3.35" not in sqlite_version:
        print("Upgrading SQLite...")

        # Step 1: Download the SQLite source tarball
        url = "https://www.sqlite.org/2024/sqlite-autoconf-3470200.tar.gz"
        tarball = os.path.join(sqlite_dir, "sqlite.tar.gz")
        print(f"Downloading SQLite from {url}...")

        result = subprocess.run(["curl", "-o", tarball, url], capture_output=True)
        if result.returncode != 0:
            print(f"Error downloading SQLite: {result.stderr.decode()}")
            return
        
        # Step 2: Extract the tarball
        print("Extracting SQLite source code...")
        result = subprocess.run(["tar", "-xzf", tarball, "-C", sqlite_dir], capture_output=True)
        if result.returncode != 0:
            print(f"Error extracting the tarball: {result.stderr.decode()}")
            return

        # Automatically detect the extracted directory
        extracted_dir = next(
            (d for d in os.listdir(sqlite_dir) if d.startswith("sqlite-autoconf-")),
            None
        )
        if not extracted_dir:
            print("Failed to detect the extracted SQLite directory!")
            return

        extracted_dir = os.path.join(sqlite_dir, extracted_dir)
        print(f"SQLite source directory: {extracted_dir}")

        # Step 3: Build SQLite
        print("Building SQLite...")
        os.chdir(extracted_dir)
        subprocess.run(["./configure", f"--prefix={sqlite_dir}"], check=True)
        subprocess.run(["make"], check=True)
        subprocess.run(["make", "install"], check=True)

        # Step 4: Verify the updated SQLite version
        sqlite_version = subprocess.getoutput(f"{sqlite_bin} --version")
        print(f"Updated SQLite version: {sqlite_version}")

    # Step 5: Rebuild Python SQLite bindings
    print("Rebuilding Python SQLite bindings...")
    subprocess.run(["pip", "uninstall", "-y", "pysqlite3"], check=True)
    subprocess.run(["pip", "install", "pysqlite3-binary"], check=True)

    # Ensure the upgraded SQLite is used
    print("Configuring environment to use upgraded SQLite...")

    # Update the environment variables to prioritize the local SQLite binary and libraries
    os.environ["PATH"] = f"{os.path.join(sqlite_dir, 'bin')}:" + os.environ["PATH"]
    os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(sqlite_dir, 'lib')}:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    # Set the PYTHONPATH to use the local `pysqlite3` package
    # This ensures the right SQLite is used when importing `sqlite3` module
    sys.path.insert(0, os.path.join(sqlite_dir, 'lib', 'python3.10', 'site-packages'))

    print(f"SQLite is now set to use the version installed at {sqlite_bin}")

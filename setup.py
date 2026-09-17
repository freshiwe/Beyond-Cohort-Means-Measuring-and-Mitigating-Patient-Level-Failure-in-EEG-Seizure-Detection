import os

# Define the base project directory
base_dir = "EEG-Seizure-Detection"

# Define the directory structure
structure = {
    base_dir: [
        "environment_setup.py",
        "environment.yml",
        "requirements.txt",
        "Dockerfile",
        "INSTALL.md",
        "README.md",
        "setup.py",
        {".github": [
            {"workflows": ["ci.yml"]}
        ]},
        "src",  # already created, but we ensure existence
        "configs",
        {"scripts": [
            "test_synthetic.py",
            "download_data.py",
            "preprocess.py",
            "train_loso.py",
            "evaluate.py",
            {"nrp": [
                "loso_job.yaml",
                "loso_array_job.yaml"
            ]}
        ]},
        "notebooks",
        {"data": ["README.md"]}
    ]
}

def create_structure(base, items):
    for item in items:
        if isinstance(item, str):
            # Create file or directory
            if "." in item:  # crude check: treat as file if it has an extension
                file_path = os.path.join(base, item)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("")  # create empty file
            else:
                os.makedirs(os.path.join(base, item), exist_ok=True)
        elif isinstance(item, dict):
            for folder, subitems in item.items():
                folder_path = os.path.join(base, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, subitems)

def main():
    create_structure(".", structure[base_dir])
    print(f"Project structure for '{base_dir}' created successfully.")

if __name__ == "__main__":
    main()

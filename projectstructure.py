import os
from pathlib import Path

file_name = "src"

list_of_files = [

    f"{file_name}/__init__.py",
    f"{file_name}/components/__init__.py",
    f"{file_name}/components/data_ingestion.py",
    f"{file_name}/components/data_validation.py",
    f"{file_name}/components/data_transformation.py",
    f"{file_name}/components/model_trainer.py",
    f"{file_name}/utils/__init__.py",
    f"{file_name}/utils/util.py",
    f"{file_name}/logging/__init__.py",
    f"{file_name}/logging/logger.py",
    f"{file_name}/exceptions/__init__.py",
    f"{file_name}/exceptions/exception.py",
    f"{file_name}/pipeline/__init__.py",
    f"{file_name}/pipeline/train_pipeline.py",
    f"{file_name}/pipeline/predict_pipeline.py",
    f"{file_name}/constants/__init__.py",
    f"{file_name}/constants/constant.py",
]

for path in list_of_files:
    filepath=Path(path)

    filedir,filename=os.path.split(path)
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass

In order to run the model use the following command serially 1. python -m venv .venv 2. .\.venv\Scripts\Activate.ps1 for powershell and .\.venv\Scripts\activate.bat for cmd 3. python -m pip install --upgrade pip 4. pip install -r requirements.txt

In order to train the model use the following command serially 1. python -m ai.train --episodes 800 --environment both 2. python main.py 3. python -m ai.evaluate --maps 100

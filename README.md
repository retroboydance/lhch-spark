This is the official repository for the LHCH spark project.

The repository will hold all scripts related to running training on the DGX spark. including, but not limited to:
  - model architecture
  - utils
  - optimiser
  - tokeniser
  - dataset (not dataset content)

To set up the python environment, create the virtual environment:
```powershell
python -m venv [name-of-environment]
```

Activate virtual environment (Windows):
```powershell
cd [name-of-environment]
.\Scripts\activate
```

Activate virtual environment (Linux):
```bash
cd [name-of-environment]
source .\bin\activate
```

```powershell
pip install -r requirements.txt
```

NOTE: patient data is not stored here. Any commits containing confidential information will be denied.
# Exploring Hugging Face models compatibility with OpenVINO

To run the code:

1. Setup the environment:
   ```bash
   bash scripts/bootstrap.sh
   ```
   
1. Run models collection:
   ```bash
   bash scripts/run.sh
   ```
   
1. Run models validation:
   ```bash
   bash scripts/run_validation.sh
   ```

1. Activate the environment:
   ```bash
   source venv/bin/activate
   ```
   
1. To monitor validation progress, or once previous step finishes, collect final validation report:
   ```bash
   python process_validation_report.py
   ```
   
1. Get summary of all errors:
   ```bash
   python rejected_report.py
   ```

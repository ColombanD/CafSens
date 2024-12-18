# Cafsens
Measuring correlation between Catastrophic Forgetting and Model Sensitivity

## Wiki
The main logic is in the `main.py`.
To run the code, one uses
```bash
python main.py --model "model_name" --old-dataset "old_data_set_name" --new-dataset "new_data_set_name"
```
To see which data sets and models are available, run 
```bash
python main.py -h
```
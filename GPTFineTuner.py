import pandas as pd
import json
import os
import time
from openai import OpenAI
from sklearn.metrics import mean_absolute_error
import numpy as np

class GPTFineTuner:
    def __init__(self, api_key, dataset_path, model_name="gpt-4.1-nano-2025-04-14", num_examples=50, jsonl_path='finetune_hea.jsonl'):
        """Initialize the fine-tuner with API key, dataset, and params."""
        self.client = OpenAI(api_key=api_key)
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.num_examples = num_examples
        self.jsonl_path = jsonl_path
        self.file_id = None
        self.job_id = None
        self.fine_tuned_model = None
        self.dataset = None
        self.holdout_data = None

    def build_dataset(self):
        """Build and save JSONL dataset from CSV."""
        self.dataset = pd.read_csv(self.dataset_path)
        train_data = self.dataset.sample(n=self.num_examples, random_state=42)
        self.holdout_data = self.dataset.drop(train_data.index).sample(n=min(10, len(self.dataset) - self.num_examples))
        
        jsonl_lines = []
        for _, row in train_data.iterrows():
            user_prompt = row['prompt']
            gibbs = row['Gibbs']
            interpretation = "strong adsorption, suggesting high stability." if gibbs < 0 else "weak adsorption, potentially unstable."
            assistant_response = f"The Gibbs free energy of nitrate adsorption is {gibbs:.4f} eV. This indicates {interpretation}"
            entry = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            jsonl_lines.append(json.dumps(entry))
        
        with open(self.jsonl_path, 'w') as f:
            for line in jsonl_lines:
                f.write(line + '\n')
        
        print(f"Built {len(jsonl_lines)} examples in {self.jsonl_path}. Holdout size: {len(self.holdout_data)}")
        return self.jsonl_path

    def upload_data(self):
        """Upload JSONL file to OpenAI."""
        if not os.path.exists(self.jsonl_path):
            raise ValueError("Run build_dataset first.")
        with open(self.jsonl_path, 'rb') as f:
            response = self.client.files.create(file=f, purpose='fine-tune')
        self.file_id = response.id
        print("Uploaded File ID:", self.file_id)
        return self.file_id

    def create_job(self):
        """Create and monitor fine-tuning job."""
        if not self.file_id:
            raise ValueError("Run upload_data first.")
        response = self.client.fine_tuning.jobs.create(
            training_file=self.file_id,
            model=self.model_name
        )
        self.job_id = response.id
        print("Job ID:", self.job_id)
        
        # Poll until complete
        while True:
            job = self.client.fine_tuning.jobs.retrieve(self.job_id)
            print(f"Status: {job.status}")
            if job.status == 'succeeded':
                self.fine_tuned_model = job.fine_tuned_model
                print("Fine-tuned Model ID:", self.fine_tuned_model)
                break
            elif job.status in ['failed', 'cancelled']:
                raise ValueError(f"Job {job.status}: {job.error}")
            time.sleep(60)  # Wait 1 min
        
        return self.fine_tuned_model

    def evaluate_model(self):
        """Evaluate fine-tuned model on holdout data."""
        if not self.fine_tuned_model:
            raise ValueError("Run create_job first.")
        predictions = []
        actuals = []
        for _, row in self.holdout_data.iterrows():
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model,
                messages=[{"role": "user", "content": row['prompt']}],
                max_tokens=50
            )
            pred_text = response.choices[0].message.content
            try:
                pred_gibbs = float(pred_text.split('is ')[1].split(' eV')[0])
            except:
                pred_gibbs = np.nan
            predictions.append(pred_gibbs)
            actuals.append(row['Gibbs'])
        
        mae = mean_absolute_error([a for a, p in zip(actuals, predictions) if not np.isnan(p)], 
                                  [p for p in predictions if not np.isnan(p)])
        print("Evaluation MAE:", mae)
        return mae

    def run_all(self):
        """Execute all steps sequentially."""
        self.build_dataset()
        self.upload_data()
        self.create_job()
        return self.evaluate_model()


os.environ['OPENAI_API_KEY'] = 'api-key-here'  
tuner = GPTFineTuner(api_key=os.getenv('OPENAI_API_KEY'), 
                     dataset_path="/home1/nhuynh2023/datasets/PDF_HEA_Gibbs/HEA_Dataset_with_embeddings.csv",
                     num_examples=10)  # Small for testing
tuner.run_all()
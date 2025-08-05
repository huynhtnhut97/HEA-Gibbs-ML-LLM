from openai import OpenAI
import os
import json

class LLMresponseder:
    def __init__(self, api_key, job_id, checkpoint_id=None, json_path=None):
        """Initialize with API key, job ID, optional checkpoint ID or JSON path."""
        self.client = OpenAI(api_key=api_key)
        self.job_id = job_id
        self.checkpoint_id = checkpoint_id
        self.model_id = None
        self.metrics = None
        if json_path:
            self.parse_json(json_path)

    def parse_json(self, json_path):
        """Parse JSON to extract IDs and metrics."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data['object'] != "fine_tuning.job.checkpoint":
            raise ValueError("Invalid JSON object type.")
        self.checkpoint_id = data['id']
        self.job_id = data['fine_tuning_job_id']
        self.model_id = data['fine_tuned_model_checkpoint']
        self.metrics = data['metrics']
        print(f"Parsed Checkpoint ID: {self.checkpoint_id}")
        print(f"Job ID: {self.job_id}")
        print(f"Model ID: {self.model_id}")
        print(f"Metrics: {self.metrics}")
        return self.model_id, self.metrics

    def retrieve_checkpoint_model(self):
        """Retrieve model ID from API if no JSON."""
        if not self.checkpoint_id:
            raise ValueError("Provide checkpoint_id or json_path.")
        ckpt = self.client.fine_tuning.jobs.checkpoints.retrieve(self.job_id, self.checkpoint_id)
        self.model_id = ckpt.fine_tuned_model_checkpoint
        self.metrics = ckpt.metrics
        print("Retrieved Model ID:", self.model_id)
        print("Metrics:", self.metrics)
        return self.model_id

    def get_response(self, input_text, max_tokens=100, temperature=0.7):
        """Get response from the model."""
        if not self.model_id:
            self.retrieve_checkpoint_model()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": input_text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            output = response.choices[0].message.content.strip()
            print(f"Response: {output}")
            return output
        except Exception as e:
            print(f"Error: {e}")
            return None

# Test with JSON
os.environ['OPENAI_API_KEY'] = 'api-key'
responder = LLMresponseder(api_key=os.getenv('OPENAI_API_KEY'), job_id="ftjob-abc123", json_path='test_checkpoint.json')

# Test response (mock ID may fail real API; comment if needed)
test_input = "Among Fe, Co, Ni, Cu, and Zn, which one can serve as a better active site for stronger nitrate adsorption?"
response = responder.get_response(test_input)
print("Test Response:", response)
import requests
import os

def download_dataset(url, filepath):
    """Download dataset from URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filepath}")
    else:
        print(f"Failed to download from {url}")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    # Download the fake job postings dataset
    url = "https://raw.githubusercontent.com/shivam5992/fake-job-posting/master/fake_job_postings.csv"
    download_dataset(url, 'data/fake_job_postings.csv')
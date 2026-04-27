import boto3
import os

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('S3_URL'),
        aws_access_key_id=os.getenv('S3_KEY'),
        aws_secret_access_key=os.getenv('S3_SECRET')
    )

def ensure_bucket():
    client = get_minio_client()
    try:
        client.head_bucket(Bucket='models')
    except:
        client.create_bucket(Bucket='models')
        print("Bucket 'models' créé")

def upload_model(path='model.pkl'):
    client = get_minio_client()
    ensure_bucket()
    client.upload_file(path, 'models', 'model.pkl')
    print('Modele pickle uploadé sur minIO')

def download_model(path='model.pkl'):
    client = get_minio_client()
    try:
        client.download_file('models', 'model.pkl', path)
        print('Modele pickle téléchargé depuis minIO')
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Le fichier model.pkl est absent de minIO")
        else:
            raise

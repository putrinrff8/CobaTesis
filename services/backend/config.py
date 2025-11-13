import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{username}:{password}@{host}/{database}'.format(
        username=os.environ.get('DB_USERNAME', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        host=os.environ.get('DB_HOST', 'localhost:3306'),
        database=os.environ.get('DB_DATABASE', 'flask-skripsi')
    )

    SECRET_KEY = 'skripsi-app'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET', 'xxxxxx')
    UPLOAD_FOLDER = "assets"
    UPLOAD_FOLDER_VIDEO = "videos"
    UPLOAD_FOLDER_IMAGE = "images"
    UPLOAD_FOLDER_MODEL = "models"
    UPLOAD_FOLDER_DATA = "data"
    MAX_VIDEO_CONTENT_LENGTH = 10 * 1024 * 1024
    WTF_CSRF_ENABLED=False

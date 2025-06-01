from app import app as application

# This file is needed for WSGI servers to find the app object
if __name__ == "__main__":
    application.run()

# social-networks-analysis-project

## Initializing/Running Steps

### Create Spotify App

1. Visit the [Spotify developer portal](https://developer.spotify.com/dashboard/) and log in with your Spotify account.

2. Click the `Create an App` button. Enter any name and an optional description. Accept terms of service and click `Create`.

3. In the new App `Overview` screen, click on `Edit Settings` and scroll down to `Redirect URIs`. Add `http://127.0.0.1:1235/callback` (or any port number you would like). Hit the `Save` button and return to the App `Overview` screen.

4. Underneath the app name and description on the lefthand side, you will see a `Show Client Secret` link. Click on it and save both the `Client Secret` and the `Client ID` (check next step to where to save these values).

### Create and populate .env file

In the root of the repository, create a ``.env`` file and add the following:
```
SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID
SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET
SPOTIFY_REDIRECT_URI=YOUR_REDIRECT_URI
```

### Create Python virtual environment

Run all commands in root of repository!

1. Create new Python virtual environment:
```
python -m venv .venv
```

2. Activate the virtual environment:
```
source .venv/bin/activate
```

3. To kill virtual environment, run
```
deactivate
```

### Install required Python packages

After the virtual environment is running, install the required packages with the following command:
```
pip install -r requirements.txt
```

### Before running, create data folders

Create data folders such that the structure of the project looks like this:
```
    social-networks-analysis-project
            |________________ .venv
            |
            |________________ data
            |                   |____ cache
            |                           |____ album-tracks
            |                           |____ artist-releases
            |                           |____ artists
            |________________ data-gathering                               
```

### Running

1. In your terminal, navigate to the `data-gathering` folder
2. Run the `data-gathering.py` script with the following command:
```
python data-gathering.py
```

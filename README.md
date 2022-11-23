# GestureDiffuser
## Installing Packages
If you don't have virtual env installed:
`pip install virtualenv`


- `cd gesture-diffuser`
- `virtualenv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

To deactivate the virtual environment
- `deactivate`

## Preprocessing Data
The following steps are used to preprocess data:

### Preprocessing Motion Data
`python3 bvh2features.py --bvh_dir ../data/allRecSample/ --dest_dir ../data/allRecFeatures`
